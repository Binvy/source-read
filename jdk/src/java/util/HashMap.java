/*
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
 * ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */

package java.util;

import sun.misc.SharedSecrets;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.Serializable;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Hash table based implementation of the <tt>Map</tt> interface.  This
 * implementation provides all of the optional map operations, and permits
 * <tt>null</tt> values and the <tt>null</tt> key.  (The <tt>HashMap</tt>
 * class is roughly equivalent to <tt>Hashtable</tt>, except that it is
 * unsynchronized and permits nulls.)  This class makes no guarantees as to
 * the order of the map; in particular, it does not guarantee that the order
 * will remain constant over time.
 *
 * <p>This implementation provides constant-time performance for the basic
 * operations (<tt>get</tt> and <tt>put</tt>), assuming the hash function
 * disperses the elements properly among the buckets.  Iteration over
 * collection views requires time proportional to the "capacity" of the
 * <tt>HashMap</tt> instance (the number of buckets) plus its size (the number
 * of key-value mappings).  Thus, it's very important not to set the initial
 * capacity too high (or the load factor too low) if iteration performance is
 * important.
 *
 * <p>An instance of <tt>HashMap</tt> has two parameters that affect its
 * performance: <i>initial capacity</i> and <i>load factor</i>.  The
 * <i>capacity</i> is the number of buckets in the hash table, and the initial
 * capacity is simply the capacity at the time the hash table is created.  The
 * <i>load factor</i> is a measure of how full the hash table is allowed to
 * get before its capacity is automatically increased.  When the number of
 * entries in the hash table exceeds the product of the load factor and the
 * current capacity, the hash table is <i>rehashed</i> (that is, internal data
 * structures are rebuilt) so that the hash table has approximately twice the
 * number of buckets.
 *
 * <p>As a general rule, the default load factor (.75) offers a good
 * tradeoff between time and space costs.  Higher values decrease the
 * space overhead but increase the lookup cost (reflected in most of
 * the operations of the <tt>HashMap</tt> class, including
 * <tt>get</tt> and <tt>put</tt>).  The expected number of entries in
 * the map and its load factor should be taken into account when
 * setting its initial capacity, so as to minimize the number of
 * rehash operations.  If the initial capacity is greater than the
 * maximum number of entries divided by the load factor, no rehash
 * operations will ever occur.
 *
 * <p>If many mappings are to be stored in a <tt>HashMap</tt>
 * instance, creating it with a sufficiently large capacity will allow
 * the mappings to be stored more efficiently than letting it perform
 * automatic rehashing as needed to grow the table.  Note that using
 * many keys with the same {@code hashCode()} is a sure way to slow
 * down performance of any hash table. To ameliorate impact, when keys
 * are {@link Comparable}, this class may use comparison order among
 * keys to help break ties.
 *
 * <p><strong>Note that this implementation is not synchronized.</strong>
 * If multiple threads access a hash map concurrently, and at least one of
 * the threads modifies the map structurally, it <i>must</i> be
 * synchronized externally.  (A structural modification is any operation
 * that adds or deletes one or more mappings; merely changing the value
 * associated with a key that an instance already contains is not a
 * structural modification.)  This is typically accomplished by
 * synchronizing on some object that naturally encapsulates the map.
 *
 * If no such object exists, the map should be "wrapped" using the
 * {@link Collections#synchronizedMap Collections.synchronizedMap}
 * method.  This is best done at creation time, to prevent accidental
 * unsynchronized access to the map:<pre>
 *   Map m = Collections.synchronizedMap(new HashMap(...));</pre>
 *
 * <p>The iterators returned by all of this class's "collection view methods"
 * are <i>fail-fast</i>: if the map is structurally modified at any time after
 * the iterator is created, in any way except through the iterator's own
 * <tt>remove</tt> method, the iterator will throw a
 * {@link ConcurrentModificationException}.  Thus, in the face of concurrent
 * modification, the iterator fails quickly and cleanly, rather than risking
 * arbitrary, non-deterministic behavior at an undetermined time in the
 * future.
 *
 * <p>Note that the fail-fast behavior of an iterator cannot be guaranteed
 * as it is, generally speaking, impossible to make any hard guarantees in the
 * presence of unsynchronized concurrent modification.  Fail-fast iterators
 * throw <tt>ConcurrentModificationException</tt> on a best-effort basis.
 * Therefore, it would be wrong to write a program that depended on this
 * exception for its correctness: <i>the fail-fast behavior of iterators
 * should be used only to detect bugs.</i>
 *
 * <p>This class is a member of the
 * <a href="{@docRoot}/../technotes/guides/collections/index.html">
 * Java Collections Framework</a>.
 *
 * @param <K> the type of keys maintained by this map
 * @param <V> the type of mapped values
 *
 * @author  Doug Lea
 * @author  Josh Bloch
 * @author  Arthur van Hoff
 * @author  Neal Gafter
 * @see     Object#hashCode()
 * @see     Collection
 * @see     Map
 * @see     TreeMap
 * @see     Hashtable
 * @since   1.2
 */
public class HashMap<K,V> extends AbstractMap<K,V>
    implements Map<K,V>, Cloneable, Serializable {

    private static final long serialVersionUID = 362498820763181265L;

    /*
     * Implementation notes.
     *
     * This map usually acts as a binned (bucketed) hash table, but
     * when bins get too large, they are transformed into bins of
     * TreeNodes, each structured similarly to those in
     * java.util.TreeMap. Most methods try to use normal bins, but
     * relay to TreeNode methods when applicable (simply by checking
     * instanceof a node).  Bins of TreeNodes may be traversed and
     * used like any others, but additionally support faster lookup
     * when overpopulated. However, since the vast majority of bins in
     * normal use are not overpopulated, checking for existence of
     * tree bins may be delayed in the course of table methods.
     *
     * Tree bins (i.e., bins whose elements are all TreeNodes) are
     * ordered primarily by hashCode, but in the case of ties, if two
     * elements are of the same "class C implements Comparable<C>",
     * type then their compareTo method is used for ordering. (We
     * conservatively check generic types via reflection to validate
     * this -- see method comparableClassFor).  The added complexity
     * of tree bins is worthwhile in providing worst-case O(log n)
     * operations when keys either have distinct hashes or are
     * orderable, Thus, performance degrades gracefully under
     * accidental or malicious usages in which hashCode() methods
     * return values that are poorly distributed, as well as those in
     * which many keys share a hashCode, so long as they are also
     * Comparable. (If neither of these apply, we may waste about a
     * factor of two in time and space compared to taking no
     * precautions. But the only known cases stem from poor user
     * programming practices that are already so slow that this makes
     * little difference.)
     *
     * Because TreeNodes are about twice the size of regular nodes, we
     * use them only when bins contain enough nodes to warrant use
     * (see TREEIFY_THRESHOLD). And when they become too small (due to
     * removal or resizing) they are converted back to plain bins.  In
     * usages with well-distributed user hashCodes, tree bins are
     * rarely used.  Ideally, under random hashCodes, the frequency of
     * nodes in bins follows a Poisson distribution
     * (http://en.wikipedia.org/wiki/Poisson_distribution) with a
     * parameter of about 0.5 on average for the default resizing
     * threshold of 0.75, although with a large variance because of
     * resizing granularity. Ignoring variance, the expected
     * occurrences of list size k are (exp(-0.5) * pow(0.5, k) /
     * factorial(k)). The first values are:
     *
     * 0:    0.60653066
     * 1:    0.30326533
     * 2:    0.07581633
     * 3:    0.01263606
     * 4:    0.00157952
     * 5:    0.00015795
     * 6:    0.00001316
     * 7:    0.00000094
     * 8:    0.00000006
     * more: less than 1 in ten million
     *
     * The root of a tree bin is normally its first node.  However,
     * sometimes (currently only upon Iterator.remove), the root might
     * be elsewhere, but can be recovered following parent links
     * (method TreeNode.root()).
     *
     * All applicable internal methods accept a hash code as an
     * argument (as normally supplied from a public method), allowing
     * them to call each other without recomputing user hashCodes.
     * Most internal methods also accept a "tab" argument, that is
     * normally the current table, but may be a new or old one when
     * resizing or converting.
     *
     * When bin lists are treeified, split, or untreeified, we keep
     * them in the same relative access/traversal order (i.e., field
     * Node.next) to better preserve locality, and to slightly
     * simplify handling of splits and traversals that invoke
     * iterator.remove. When using comparators on insertion, to keep a
     * total ordering (or as close as is required here) across
     * rebalancings, we compare classes and identityHashCodes as
     * tie-breakers.
     *
     * The use and transitions among plain vs tree modes is
     * complicated by the existence of subclass LinkedHashMap. See
     * below for hook methods defined to be invoked upon insertion,
     * removal and access that allow LinkedHashMap internals to
     * otherwise remain independent of these mechanics. (This also
     * requires that a map instance be passed to some utility methods
     * that may create new nodes.)
     *
     * The concurrent-programming-like SSA-based coding style helps
     * avoid aliasing errors amid all of the twisty pointer operations.
     */

    /**
     * The default initial capacity - MUST be a power of two.
     */
    static final int DEFAULT_INITIAL_CAPACITY = 1 << 4; // aka 16 // 默认容量16

    /**
     * The maximum capacity, used if a higher value is implicitly specified
     * by either of the constructors with arguments.
     * MUST be a power of two <= 1<<30.
     */
    static final int MAXIMUM_CAPACITY = 1 << 30; // 最大容量

    /**
     * The load factor used when none specified in constructor.
     */
    static final float DEFAULT_LOAD_FACTOR = 0.75f; // 默认负载因子0.75

    /**
     * The bin count threshold for using a tree rather than list for a
     * bin.  Bins are converted to trees when adding an element to a
     * bin with at least this many nodes. The value must be greater
     * than 2 and should be at least 8 to mesh with assumptions in
     * tree removal about conversion back to plain bins upon
     * shrinkage.
     */
    static final int TREEIFY_THRESHOLD = 8; // 链表节点转为红黑树节点的阈值，9个节点转

    /**
     * The bin count threshold for untreeifying a (split) bin during a
     * resize operation. Should be less than TREEIFY_THRESHOLD, and at
     * most 6 to mesh with shrinkage detection under removal.
     */
    static final int UNTREEIFY_THRESHOLD = 6; // 红黑树节点转为链表节点的阈值，6个节点转

    /**
     * The smallest table capacity for which bins may be treeified.
     * (Otherwise the table is resized if too many nodes in a bin.)
     * Should be at least 4 * TREEIFY_THRESHOLD to avoid conflicts
     * between resizing and treeification thresholds.
     */
    static final int MIN_TREEIFY_CAPACITY = 64; // 链表转红黑树时，table数组的最小长度

    /**
     * Basic hash bin node, used for most entries.  (See below for
     * TreeNode subclass, and in LinkedHashMap for its Entry subclass.)
     */
    static class Node<K,V> implements Map.Entry<K,V> { // 基本hash节点
        final int hash;
        final K key;
        V value;
        Node<K,V> next;

        Node(int hash, K key, V value, Node<K,V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }

        public final K getKey()        { return key; }
        public final V getValue()      { return value; }
        public final String toString() { return key + "=" + value; }

        public final int hashCode() {
            return Objects.hashCode(key) ^ Objects.hashCode(value);
        }

        public final V setValue(V newValue) {
            V oldValue = value;
            value = newValue;
            return oldValue;
        }

        public final boolean equals(Object o) {
            if (o == this)
                return true;
            if (o instanceof Map.Entry) {
                Map.Entry<?,?> e = (Map.Entry<?,?>)o;
                if (Objects.equals(key, e.getKey()) &&
                    Objects.equals(value, e.getValue()))
                    return true;
            }
            return false;
        }
    }

    /* ---------------- Static utilities -------------- */

    /**
     * Computes key.hashCode() and spreads (XORs) higher bits of hash
     * to lower.  Because the table uses power-of-two masking, sets of
     * hashes that vary only in bits above the current mask will
     * always collide. (Among known examples are sets of Float keys
     * holding consecutive whole numbers in small tables.)  So we
     * apply a transform that spreads the impact of higher bits
     * downward. There is a tradeoff between speed, utility, and
     * quality of bit-spreading. Because many common sets of hashes
     * are already reasonably distributed (so don't benefit from
     * spreading), and because we use trees to handle large sets of
     * collisions in bins, we just XOR some shifted bits in the
     * cheapest possible way to reduce systematic lossage, as well as
     * to incorporate impact of the highest bits that would otherwise
     * never be used in index calculations because of table bounds.
     */
    static final int hash(Object key) {
        // 方法：
        //     1. 先拿到key的hashCode值，
        //     2. 将hashCode的高16位参与运算
        // 目的：
        //       主要是为了在table的length小的时候，让高位也参与运算，
        //       高16位 ^ 低16位，信息混合，使得新值更具有随机性，并且不会有太大的开销
        // 原因：
        //    1. HashMap的实现方式依赖于hashCode函数的实现，特别是，hash值的低位应该均匀分布，
        //       如果在较低位上有许多冲突，则HashMap将会出现较大的桶碰撞几率
        //    2. 因为HashCode方法的实现超出了HashMap的控制范围（每个对象都可以有自己的实现方式，质量参差不齐）
        int h;
        return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    }

    /**
     * Returns x's Class if it is of the form "class C implements
     * Comparable<C>", else null.
     * 如果x实现了Comparable接口，则返回x的Class
     */
    static Class<?> comparableClassFor(Object x) {
        if (x instanceof Comparable) {
            Class<?> c; Type[] ts, as; Type t; ParameterizedType p;
            if ((c = x.getClass()) == String.class) // bypass checks
                return c;
            if ((ts = c.getGenericInterfaces()) != null) {
                for (int i = 0; i < ts.length; ++i) {
                    if (((t = ts[i]) instanceof ParameterizedType) &&
                        ((p = (ParameterizedType)t).getRawType() ==
                         Comparable.class) &&
                        (as = p.getActualTypeArguments()) != null &&
                        as.length == 1 && as[0] == c) // type arg is c
                        return c;
                }
            }
        }
        return null;
    }

    /**
     * Returns k.compareTo(x) if x matches kc (k's screened comparable
     * class), else 0.
     */
    @SuppressWarnings({"rawtypes","unchecked"}) // for cast to Comparable
    static int compareComparables(Class<?> kc, Object k, Object x) {
        return (x == null || x.getClass() != kc ? 0 :
                ((Comparable)k).compareTo(x));
    }

    /**
     * Returns a power of two size for the given target capacity.
     */
    static final int tableSizeFor(int cap) {
        int n = cap - 1; // 当cap为2的n次方时，返回自己（否则，当cap为2的n次方时，最后返回的为cap*2）
        n |= n >>> 1; // 01xxxx...xxx |= 001xxx...xxx => 011xxx...xxx
        n |= n >>> 2; // 011xxx...xxx |= 01111x...xxx => 01111x...xxx
        n |= n >>> 4; // 01111x...xxx |= 01111x...xxx => 011111...xxx
        n |= n >>> 8; // 01111x...xxx |= 01111x...xxx => 011111...xxx
        n |= n >>> 16; // 01111x...xxx |= 011111...111 => 011111...111
        return (n < 0) ? 1 : (n >= MAXIMUM_CAPACITY) ? MAXIMUM_CAPACITY : n + 1; // n + 1 = 100000...000
    }

    /* ---------------- Fields -------------- */

    /**
     * The table, initialized on first use, and resized as
     * necessary. When allocated, length is always a power of two.
     * (We also tolerate length zero in some operations to allow
     * bootstrapping mechanics that are currently not needed.)
     */
    transient Node<K,V>[] table; // Node数组的形式

    /**
     * Holds cached entrySet(). Note that AbstractMap fields are used
     * for keySet() and values().
     */
    transient Set<Map.Entry<K,V>> entrySet;

    /**
     * The number of key-value mappings contained in this map.
     */
    transient int size; // map中包含的键值对的数量

    /**
     * The number of times this HashMap has been structurally modified
     * Structural modifications are those that change the number of mappings in
     * the HashMap or otherwise modify its internal structure (e.g.,
     * rehash).  This field is used to make iterators on Collection-views of
     * the HashMap fail-fast.  (See ConcurrentModificationException).
     */
    transient int modCount; // map被结构化变更的次数

    /**
     * The next size value at which to resize (capacity * load factor).
     *
     * @serial
     */
    // (The javadoc description is true upon serialization.
    // Additionally, if the table array has not been allocated, this
    // field holds the initial array capacity, or zero signifying
    // DEFAULT_INITIAL_CAPACITY.)
    int threshold; // map重新分配尺寸的阈值，比如：threshold = 16 * 0.75 = 12，当增加到13个键值对时进行扩容，

    /**
     * The load factor for the hash table.
     *
     * @serial
     */
    final float loadFactor; // 加载因子

    /* ---------------- Public operations -------------- */

    /**
     * Constructs an empty <tt>HashMap</tt> with the specified initial
     * capacity and load factor.
     *
     * @param  initialCapacity the initial capacity
     * @param  loadFactor      the load factor
     * @throws IllegalArgumentException if the initial capacity is negative
     *         or the load factor is nonpositive
     */
    public HashMap(int initialCapacity, float loadFactor) {
        if (initialCapacity < 0)
            throw new IllegalArgumentException("Illegal initial capacity: " +
                                               initialCapacity);
        if (initialCapacity > MAXIMUM_CAPACITY)
            initialCapacity = MAXIMUM_CAPACITY;
        if (loadFactor <= 0 || Float.isNaN(loadFactor))
            throw new IllegalArgumentException("Illegal load factor: " +
                                               loadFactor);
        this.loadFactor = loadFactor;
        this.threshold = tableSizeFor(initialCapacity);
    }

    /**
     * Constructs an empty <tt>HashMap</tt> with the specified initial
     * capacity and the default load factor (0.75).
     *
     * @param  initialCapacity the initial capacity.
     * @throws IllegalArgumentException if the initial capacity is negative.
     */
    public HashMap(int initialCapacity) {
        this(initialCapacity, DEFAULT_LOAD_FACTOR);
    }

    /**
     * Constructs an empty <tt>HashMap</tt> with the default initial capacity
     * (16) and the default load factor (0.75).
     */
    public HashMap() {
        this.loadFactor = DEFAULT_LOAD_FACTOR; // all other fields defaulted
    }

    /**
     * Constructs a new <tt>HashMap</tt> with the same mappings as the
     * specified <tt>Map</tt>.  The <tt>HashMap</tt> is created with
     * default load factor (0.75) and an initial capacity sufficient to
     * hold the mappings in the specified <tt>Map</tt>.
     *
     * @param   m the map whose mappings are to be placed in this map
     * @throws  NullPointerException if the specified map is null
     */
    public HashMap(Map<? extends K, ? extends V> m) {
        this.loadFactor = DEFAULT_LOAD_FACTOR;
        putMapEntries(m, false);
    }

    /**
     * Implements Map.putAll and Map constructor
     *
     * @param m the map
     * @param evict false when initially constructing this map, else
     * true (relayed to method afterNodeInsertion).
     */
    final void putMapEntries(Map<? extends K, ? extends V> m, boolean evict) {
        int s = m.size();
        if (s > 0) {
            if (table == null) { // pre-size
                float ft = ((float)s / loadFactor) + 1.0F;
                int t = ((ft < (float)MAXIMUM_CAPACITY) ?
                         (int)ft : MAXIMUM_CAPACITY);
                if (t > threshold)
                    threshold = tableSizeFor(t);
            }
            else if (s > threshold)
                resize();
            for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
                K key = e.getKey();
                V value = e.getValue();
                putVal(hash(key), key, value, false, evict);
            }
        }
    }

    /**
     * Returns the number of key-value mappings in this map.
     *
     * @return the number of key-value mappings in this map
     */
    public int size() {
        return size;
    }

    /**
     * Returns <tt>true</tt> if this map contains no key-value mappings.
     *
     * @return <tt>true</tt> if this map contains no key-value mappings
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or {@code null} if this map contains no mapping for the key.
     *
     * <p>More formally, if this map contains a mapping from a key
     * {@code k} to a value {@code v} such that {@code (key==null ? k==null :
     * key.equals(k))}, then this method returns {@code v}; otherwise
     * it returns {@code null}.  (There can be at most one such mapping.)
     *
     * <p>A return value of {@code null} does not <i>necessarily</i>
     * indicate that the map contains no mapping for the key; it's also
     * possible that the map explicitly maps the key to {@code null}.
     * The {@link #containsKey containsKey} operation may be used to
     * distinguish these two cases.
     *
     * @see #put(Object, Object)
     */
    public V get(Object key) {
        Node<K,V> e;
        return (e = getNode(hash(key), key)) == null ? null : e.value;
    }

    /**
     * Implements Map.get and related methods
     *
     * @param hash hash for key
     * @param key the key
     * @return the node, or null if none
     */
    final Node<K,V> getNode(int hash, Object key) {
        Node<K,V>[] tab; Node<K,V> first, e; int n; K k;
        if ((tab = table) != null && (n = tab.length) > 0 &&
            (first = tab[(n - 1) & hash]) != null) { // x % 2^n = x & (2^n - 1)，元素分布相对比较均匀，且&比%效率更高
            if (first.hash == hash && // always check first node
                ((k = first.key) == key || (key != null && key.equals(k))))
                return first;
            if ((e = first.next) != null) {
                // 如果是红黑树节点，调用红黑树的查找目标节点方法getTreeNode()
                if (first instanceof TreeNode)
                    return ((TreeNode<K,V>)first).getTreeNode(hash, key);
                // 走到这代表节点是链表节点，向下遍历链表，直到找到节点的key和传入的key相等时，返回该节点
                do {
                    if (e.hash == hash &&
                        ((k = e.key) == key || (key != null && key.equals(k))))
                        return e;
                } while ((e = e.next) != null);
            }
        }
        return null;
    }
    /***************************************************************************************************************
     *                            定位哈希桶数组索引位置：first = hash & (table.length - 1)                        *
     * *************************************************************************************************************
     *                                                                                                             *
     * 因为 x % 2^n = x & (2^n - 1)，所以((n = tab.length) - 1) & hash  等价于 hash % tab.length                   *
     * 例子：table.length = 16                                                                                     *
     *      h = hashCode():            1111 1111 1111 1111 1010 0000 1111 1010           --|                       *
     *      h >>> 16:                  0000 0000 0000 0000 1111 1111 1111 1111              >> hash()计算哈希值    *
     *      h ^ h >>> 16:              1111 1111 1111 1111 0101 1111 0000 0101           --|                       *
     *      table.length - 1:          0000 0000 0000 0000 0000 0000 0000 1111 = 15         >> 计算索引位置        *
     *      hash & (table.length - 1): 0000 0000 0000 0000 0000 0000 0000 0101 = 5       --|                       *
     *                                                                                                             *
     * ************************************************************************************************************/

    /**
     * Returns <tt>true</tt> if this map contains a mapping for the
     * specified key.
     *
     * @param   key   The key whose presence in this map is to be tested
     * @return <tt>true</tt> if this map contains a mapping for the specified
     * key.
     */
    public boolean containsKey(Object key) {
        return getNode(hash(key), key) != null;
    }

    /**
     * Associates the specified value with the specified key in this map.
     * If the map previously contained a mapping for the key, the old
     * value is replaced.
     *
     * @param key key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     * @return the previous value associated with <tt>key</tt>, or
     *         <tt>null</tt> if there was no mapping for <tt>key</tt>.
     *         (A <tt>null</tt> return can also indicate that the map
     *         previously associated <tt>null</tt> with <tt>key</tt>.)
     */
    public V put(K key, V value) {
        return putVal(hash(key), key, value, false, true);
    }

    /**
     * Implements Map.put and related methods
     *
     * @param hash hash for key
     * @param key the key
     * @param value the value to put
     * @param onlyIfAbsent if true, don't change existing value
     * @param evict if false, the table is in creation mode.
     * @return previous value, or null if none
     */
    final V putVal(int hash, K key, V value, boolean onlyIfAbsent,
                   boolean evict) {
        Node<K,V>[] tab; Node<K,V> p; int n, i;
        // 如果table为空，或者length为0，则调用resize方法进行初始化
        if ((tab = table) == null || (n = tab.length) == 0)
            n = (tab = resize()).length;
        // 通过hash值计算索引位置，如果table该索引位置节点为空，则新建节点
        if ((p = tab[i = (n - 1) & hash]) == null)
            tab[i] = newNode(hash, key, value, null);
        // 如果table表该索引位置不为空
        else {
            Node<K,V> e; K k;
            // 判断p节点的hash、key与传入的hash、key是否相等
            if (p.hash == hash &&
                ((k = p.key) == key || (key != null && key.equals(k))))
                // 如果相等，则p节点为要查找的目标节点，赋值给e
                e = p;
            // 判断p节点是否为红黑树节点，如果是则调用红黑树的putTreeVal方法查找目标节点
            else if (p instanceof TreeNode)
                e = ((TreeNode<K,V>)p).putTreeVal(this, tab, hash, key, value);
            // 走到这说明p节点为普通链表节点
            else {
                // 遍历此链表，binCount用于统计节点数
                for (int binCount = 0; ; ++binCount) {
                    // 判断p.next是否为空，如果为空，代表不存在目标节点，则新增一个节点插入链表尾部，并赋值给节点e
                    if ((e = p.next) == null) {
                        p.next = newNode(hash, key, value, null);
                        // 计算节点是否超过8个，减1是因为循环是从p.next开始的
                        if (binCount >= TREEIFY_THRESHOLD - 1) // -1 for 1st
                            treeifyBin(tab, hash); // 如果超过8个，调用treefiyBin方法将该链表转为红黑树
                        break;
                    }
                    // 如果e节点的hash、key与传入的hash、key相等，则e为目标节点，跳出循环，
                    if (e.hash == hash &&
                        ((k = e.key) == key || (key != null && key.equals(k))))
                        break;
                    p = e;
                }
            }
            // e不为空，代表根据传入的hash、key查找到了节点，将该节点的value值覆盖，返回旧值oldValue
            if (e != null) { // existing mapping for key
                V oldValue = e.value;
                if (!onlyIfAbsent || oldValue == null)
                    e.value = value;
                afterNodeAccess(e); // 用于LinkedHashMap
                return oldValue;
            }
        }
        ++modCount;
        if (++size > threshold) // 插入节点后超过“重新分配尺寸的阈值”，则调用resize方法进行扩容
            resize();
        afterNodeInsertion(evict); // 用于LinkedList
        return null;
    }

    /**
     * Initializes or doubles table size.  If null, allocates in
     * accord with initial capacity target held in field threshold.
     * Otherwise, because we are using power-of-two expansion, the
     * elements from each bin must either stay at same index, or move
     * with a power of two offset in the new table.
     * <pre>
     * 如果老表的容量大于0，判断老表的容量是否超过最大容量值：如果超过则将阈值设置为Integer.MAX_VALUE，并直接返回老表（此时oldCap * 2比Integer.MAX_VALUE大，因此无法进行重新分布，只是单纯的将阈值扩容到最大）；如果容量 * 2小于最大容量并且不小于16，则将阈值设置为原来的两倍。
     * 如果老表的容量为0，老表的阈值大于0，这种情况是传了容量的new方法创建的空表，将新表的容量设置为老表的阈值（这种情况发生在新创建的HashMap第一次put时，该HashMap初始化的时候传了初始容量，由于HashMap并没有capacity变量来存放容量值，因此传进来的初始容量是存放在threshold变量上（查看HashMap(int initialCapacity, float loadFactor)方法），因此此时老表的threshold的值就是我们要新创建的HashMap的capacity，所以将新表的容量设置为老表的阈值。
     * 如果老表的容量为0，老表的阈值为0，这种情况是没有传容量的new方法创建的空表，将阈值和容量设置为默认值。
     * 如果新表的阈值为空，则通过新的容量 * 负载因子获得阈值（这种情况是初始化的时候传了初始容量，跟第2点相同情况，也只有走到第2点才会走到该情况）。
     * 将当前阈值设置为刚计算出来的新的阈值，定义新表，容量为刚计算出来的新容量，将当前的表设置为新定义的表。
     * 如果老表不为空，则需遍历所有节点，将节点赋值给新表。
     * 将老表上索引为j的头结点赋值给e节点，并将老表上索引为j的节点设置为空。
     * 如果e的next节点为空，则代表老表的该位置只有1个节点，通过hash值计算新表的索引位置，直接将该节点放在新表的该位置上。
     * 如果e的next节点不为空，并且e为TreeNode，则调用split方法（见下文代码块10）进行hash分布。
     * 如果e的next节点不为空，并且e为普通的链表节点，则进行普通的hash分布。
     * 如果e的hash值与老表的容量（为一串只有1个为2的二进制数，例如16为0000 0000 0001 0000）进行位与运算为0，则说明e节点扩容后的索引位置跟老表的索引位置一样（见例子1），进行链表拼接操作：如果loTail为空，代表该节点为第一个节点，则将loHead赋值为该节点；否则将节点添加在loTail后面，并将loTail赋值为新增的节点。
     * 如果e的hash值与老表的容量（为一串只有1个为2的二进制数，例如16为0000 0000 0001 0000）进行位与运算为1，则说明e节点扩容后的索引位置为：老表的索引位置＋oldCap（见例子1），进行链表拼接操作：如果hiTail为空，代表该节点为第一个节点，则将hiHead赋值为该节点；否则将节点添加在hiTail后面，并将hiTail赋值为新增的节点。
     * 老表节点重新hash分布在新表结束后，如果loTail不为空（说明老表的数据有分布到新表上原索引位置的节点），则将最后一个节点的next设为空，并将新表上原索引位置的节点设置为对应的头结点；如果hiTail不为空（说明老表的数据有分布到新表上原索引+oldCap位置的节点），则将最后一个节点的next设为空，并将新表上索引位置为原索引+oldCap的节点设置为对应的头结点。
     * 返回新表。
     * </pre>
     * @return the table
     */
    final Node<K,V>[] resize() {
        Node<K,V>[] oldTab = table;
        int oldCap = (oldTab == null) ? 0 : oldTab.length;
        int oldThr = threshold;
        int newCap, newThr = 0;
        // 如果老table不为空
        if (oldCap > 0) {
            // 如果容量 >= 最大容量2^30，设置扩容阈值为最大int值2^31-1，然后返回
            if (oldCap >= MAXIMUM_CAPACITY) {
                threshold = Integer.MAX_VALUE;
                return oldTab;
            }
            // 如果容量*2 <= 最大容量2^30，并且容量 >= 初始容量16，则将容量、阈值设置为原来的2倍
            else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                     oldCap >= DEFAULT_INITIAL_CAPACITY)
                newThr = oldThr << 1; // double threshold
        }
        // 老表的容量为0，阈值大于0，是因为初始容量被放入阈值
        else if (oldThr > 0) // initial capacity was placed in threshold
            newCap = oldThr; // 则将新表的容量设置为老表的阈值
        // 如果老表的容量、阈值均为0，则为空表，设置默认容量和阈值
        else {               // zero initial threshold signifies using defaults
            newCap = DEFAULT_INITIAL_CAPACITY;
            newThr = (int)(DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
        }
        // 如果新表的阈值为0，则根据新的容量 * 负载因子获得阈值
        if (newThr == 0) {
            float ft = (float)newCap * loadFactor;
            newThr = (newCap < MAXIMUM_CAPACITY && ft < (float)MAXIMUM_CAPACITY ?
                      (int)ft : Integer.MAX_VALUE);
        }
        threshold = newThr; // 将当前阈值设置为刚计算得到的新阈值
        @SuppressWarnings({"rawtypes","unchecked"})
            // 定义新表，容量为刚计算得到的新容量
            Node<K,V>[] newTab = (Node<K,V>[])new Node[newCap];
        table = newTab; // 将当前的表赋值为新定义的表
        // 如果老表不为空，则需遍历节点赋值给新表
        if (oldTab != null) {
            for (int j = 0; j < oldCap; ++j) {
                Node<K,V> e;
                if ((e = oldTab[j]) != null) { // 将索引为j的老表头节点赋值给e节点
                    oldTab[j] = null; // 将老表的头节点设置为空，便于垃圾回收器回收空间
                    // 如果老表头节点的next节点为空，表示老表该索引位置只有1个节点
                    if (e.next == null)
                        newTab[e.hash & (newCap - 1)] = e; // 通过hash值计算新表的索引位置，直接将e节点放在该索引位置
                    // 如果e节点为树节点，调用树节点的hash分布（跟下面最后一个else的内容几乎相同）
                    else if (e instanceof TreeNode)
                        ((TreeNode<K,V>)e).split(this, newTab, j, oldCap);
                    else { // preserve order
                        Node<K,V> loHead = null, loTail = null; // 存储跟原索引位置相同的节点
                        Node<K,V> hiHead = null, hiTail = null; // 存储索引位置为原索引+原容量的节点
                        Node<K,V> next;
                        do {
                            next = e.next;
                            // 如果e的hash值与老表容量进行与运算为0，则扩容后的索引位置跟老表索引位置一样
                            if ((e.hash & oldCap) == 0) {
                                if (loTail == null) // 如果loTail为空，代表该节点为第一个节点
                                    loHead = e; // 则将loHead赋值给第一个节点
                                else
                                    loTail.next = e; // 否则将节点添加在loTail后面
                                loTail = e; // 并将loTail赋值为新增的节点e
                            }
                            // 如果e的hash值与老表容量进行与运算不为0，则扩容后的索引位置=老表索引位置+老表容量oldCap
                            else {
                                if (hiTail == null) // 如果hiTail为空，代表该节点为第一个节点
                                    hiHead = e; // 则将hiHead赋值给第一个节点
                                else
                                    hiTail.next = e; // 否则将节点添加在hiTail后面
                                hiTail = e; // 并将hiTail赋值为新增的节点e
                            }
                        } while ((e = next) != null);
                        if (loTail != null) {
                            loTail.next = null; // 最后一个节点的next设置为空
                            newTab[j] = loHead; // 将原索引位置的节点设置为对应的头节点
                        }
                        if (hiTail != null) {
                            hiTail.next = null; // 最后一个节点的next设置为空;
                            newTab[j + oldCap] = hiHead; // 将索引位置为原索引+oldCap的节点设置为对应的头节点
                        }
                    }
                }
            }
        }
        return newTab;
    }

    /**
     * Replaces all linked nodes in bin at index for given hash unless
     * table is too small, in which case resizes instead.
     */
    final void treeifyBin(Node<K,V>[] tab, int hash) {
        int n, index; Node<K,V> e;
        // table为空，或者length小于64，进行尺寸调整
        if (tab == null || (n = tab.length) < MIN_TREEIFY_CAPACITY)
            resize();
        // 根据hash值计算索引位置，遍历该索引位置的链表
        else if ((e = tab[index = (n - 1) & hash]) != null) {
            TreeNode<K,V> hd = null, tl = null;
            do {
                TreeNode<K,V> p = replacementTreeNode(e, null); // 链表节点转换为红黑树节点
                if (tl == null) // 第一次循环，hd设置为头节点p
                    hd = p;
                else {
                    p.prev = tl; // 当前节点的prev节点设置为上一个节点
                    tl.next = p; // 上一个节点的next节点设置为当前节点
                }
                tl = p; // tl赋值给p，在下一次循环中作为上一个节点
            } while ((e = e.next) != null);
            // 将table该索引位置赋值为新转的树节点的头节点
            if ((tab[index] = hd) != null)
                hd.treeify(tab); // 以头节点为根节点，构建红黑树
        }
    }

    /**
     * Copies all of the mappings from the specified map to this map.
     * These mappings will replace any mappings that this map had for
     * any of the keys currently in the specified map.
     *
     * @param m mappings to be stored in this map
     * @throws NullPointerException if the specified map is null
     */
    public void putAll(Map<? extends K, ? extends V> m) {
        putMapEntries(m, true);
    }

    /**
     * Removes the mapping for the specified key from this map if present.
     *
     * @param  key key whose mapping is to be removed from the map
     * @return the previous value associated with <tt>key</tt>, or
     *         <tt>null</tt> if there was no mapping for <tt>key</tt>.
     *         (A <tt>null</tt> return can also indicate that the map
     *         previously associated <tt>null</tt> with <tt>key</tt>.)
     */
    public V remove(Object key) {
        Node<K,V> e;
        return (e = removeNode(hash(key), key, null, false, true)) == null ?
            null : e.value;
    }

    /**
     * Implements Map.remove and related methods
     *
     * @param hash hash for key
     * @param key the key
     * @param value the value to match if matchValue, else ignored
     * @param matchValue if true only remove if value is equal
     * @param movable if false do not move other nodes while removing
     * @return the node, or null if none
     */
    final Node<K,V> removeNode(int hash, Object key, Object value,
                               boolean matchValue, boolean movable) {
        Node<K,V>[] tab; Node<K,V> p; int n, index;
        // 遍历查找表中与key值相等的节点
        if ((tab = table) != null && (n = tab.length) > 0 &&
            (p = tab[index = (n - 1) & hash]) != null) { // 如果table表不为空，长度不为0，头节点不为空
            Node<K,V> node = null, e; K k; V v;
            // 如果头节点p.key与入参key相同，则将头节点p赋值给node
            if (p.hash == hash &&
                ((k = p.key) == key || (key != null && key.equals(k))))
                node = p;
            // 如果头节点p.next不为空
            else if ((e = p.next) != null) {
                // 如果头节点p为红黑树节点，调用红黑树节点的查找方法
                if (p instanceof TreeNode)
                    node = ((TreeNode<K,V>)p).getTreeNode(hash, key);
                // 否则，头节点为链表节点，遍历节点查找与入参key相等的节点node
                else {
                    do {
                        if (e.hash == hash &&
                            ((k = e.key) == key ||
                             (key != null && key.equals(k)))) {
                            node = e;
                            break;
                        }
                        p = e;
                    } while ((e = e.next) != null);
                }
            }
            // 如果能查找到与传入key相同的节点node，则移除该节点
            if (node != null && (!matchValue || (v = node.value) == value ||
                                 (value != null && value.equals(v)))) {
                // 如果node节点为红黑树节点，调用红黑树节点的移除节点方法
                if (node instanceof TreeNode)
                    ((TreeNode<K,V>)node).removeTreeNode(this, tab, movable);
                // node节点==p节点，表示node节点为table头节点
                else if (node == p)
                    tab[index] = node.next; // 设置头节点为node节点的next节点
                // 否则，p节点为node的父节点
                else
                    p.next = node.next; // 设置p节点的next节点为node的next节点
                ++modCount; // 修改次数加一
                --size; // table的节点总数减一
                afterNodeRemoval(node); // 供LinkedHashMap使用
                return node; // 返回被移除的节点
            }
        }
        return null;
    }

    /**
     * Removes all of the mappings from this map.
     * The map will be empty after this call returns.
     */
    public void clear() {
        Node<K,V>[] tab;
        modCount++;
        if ((tab = table) != null && size > 0) {
            size = 0;
            for (int i = 0; i < tab.length; ++i)
                tab[i] = null;
        }
    }

    /**
     * Returns <tt>true</tt> if this map maps one or more keys to the
     * specified value.
     *
     * @param value value whose presence in this map is to be tested
     * @return <tt>true</tt> if this map maps one or more keys to the
     *         specified value
     */
    public boolean containsValue(Object value) {
        Node<K,V>[] tab; V v;
        if ((tab = table) != null && size > 0) {
            for (int i = 0; i < tab.length; ++i) {
                for (Node<K,V> e = tab[i]; e != null; e = e.next) {
                    if ((v = e.value) == value ||
                        (value != null && value.equals(v)))
                        return true;
                }
            }
        }
        return false;
    }

    /**
     * Returns a {@link Set} view of the keys contained in this map.
     * The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  If the map is modified
     * while an iteration over the set is in progress (except through
     * the iterator's own <tt>remove</tt> operation), the results of
     * the iteration are undefined.  The set supports element removal,
     * which removes the corresponding mapping from the map, via the
     * <tt>Iterator.remove</tt>, <tt>Set.remove</tt>,
     * <tt>removeAll</tt>, <tt>retainAll</tt>, and <tt>clear</tt>
     * operations.  It does not support the <tt>add</tt> or <tt>addAll</tt>
     * operations.
     *
     * @return a set view of the keys contained in this map
     */
    public Set<K> keySet() {
        Set<K> ks = keySet;
        if (ks == null) {
            ks = new KeySet();
            keySet = ks;
        }
        return ks;
    }

    final class KeySet extends AbstractSet<K> {
        public final int size()                 { return size; }
        public final void clear()               { HashMap.this.clear(); }
        public final Iterator<K> iterator()     { return new KeyIterator(); }
        public final boolean contains(Object o) { return containsKey(o); }
        public final boolean remove(Object key) {
            return removeNode(hash(key), key, null, false, true) != null;
        }
        public final Spliterator<K> spliterator() {
            return new KeySpliterator<>(HashMap.this, 0, -1, 0, 0);
        }
        public final void forEach(Consumer<? super K> action) {
            Node<K,V>[] tab;
            if (action == null)
                throw new NullPointerException();
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (int i = 0; i < tab.length; ++i) {
                    for (Node<K,V> e = tab[i]; e != null; e = e.next)
                        action.accept(e.key);
                }
                if (modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }
    }

    /**
     * Returns a {@link Collection} view of the values contained in this map.
     * The collection is backed by the map, so changes to the map are
     * reflected in the collection, and vice-versa.  If the map is
     * modified while an iteration over the collection is in progress
     * (except through the iterator's own <tt>remove</tt> operation),
     * the results of the iteration are undefined.  The collection
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <tt>Iterator.remove</tt>,
     * <tt>Collection.remove</tt>, <tt>removeAll</tt>,
     * <tt>retainAll</tt> and <tt>clear</tt> operations.  It does not
     * support the <tt>add</tt> or <tt>addAll</tt> operations.
     *
     * @return a view of the values contained in this map
     */
    public Collection<V> values() {
        Collection<V> vs = values;
        if (vs == null) {
            vs = new Values();
            values = vs;
        }
        return vs;
    }

    final class Values extends AbstractCollection<V> {
        public final int size()                 { return size; }
        public final void clear()               { HashMap.this.clear(); }
        public final Iterator<V> iterator()     { return new ValueIterator(); }
        public final boolean contains(Object o) { return containsValue(o); }
        public final Spliterator<V> spliterator() {
            return new ValueSpliterator<>(HashMap.this, 0, -1, 0, 0);
        }
        public final void forEach(Consumer<? super V> action) {
            Node<K,V>[] tab;
            if (action == null)
                throw new NullPointerException();
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (int i = 0; i < tab.length; ++i) {
                    for (Node<K,V> e = tab[i]; e != null; e = e.next)
                        action.accept(e.value);
                }
                if (modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }
    }

    /**
     * Returns a {@link Set} view of the mappings contained in this map.
     * The set is backed by the map, so changes to the map are
     * reflected in the set, and vice-versa.  If the map is modified
     * while an iteration over the set is in progress (except through
     * the iterator's own <tt>remove</tt> operation, or through the
     * <tt>setValue</tt> operation on a map entry returned by the
     * iterator) the results of the iteration are undefined.  The set
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <tt>Iterator.remove</tt>,
     * <tt>Set.remove</tt>, <tt>removeAll</tt>, <tt>retainAll</tt> and
     * <tt>clear</tt> operations.  It does not support the
     * <tt>add</tt> or <tt>addAll</tt> operations.
     *
     * @return a set view of the mappings contained in this map
     */
    public Set<Map.Entry<K,V>> entrySet() {
        Set<Map.Entry<K,V>> es;
        return (es = entrySet) == null ? (entrySet = new EntrySet()) : es;
    }

    final class EntrySet extends AbstractSet<Map.Entry<K,V>> {
        public final int size()                 { return size; }
        public final void clear()               { HashMap.this.clear(); }
        public final Iterator<Map.Entry<K,V>> iterator() {
            return new EntryIterator();
        }
        public final boolean contains(Object o) {
            if (!(o instanceof Map.Entry))
                return false;
            Map.Entry<?,?> e = (Map.Entry<?,?>) o;
            Object key = e.getKey();
            Node<K,V> candidate = getNode(hash(key), key);
            return candidate != null && candidate.equals(e);
        }
        public final boolean remove(Object o) {
            if (o instanceof Map.Entry) {
                Map.Entry<?,?> e = (Map.Entry<?,?>) o;
                Object key = e.getKey();
                Object value = e.getValue();
                return removeNode(hash(key), key, value, true, true) != null;
            }
            return false;
        }
        public final Spliterator<Map.Entry<K,V>> spliterator() {
            return new EntrySpliterator<>(HashMap.this, 0, -1, 0, 0);
        }
        public final void forEach(Consumer<? super Map.Entry<K,V>> action) {
            Node<K,V>[] tab;
            if (action == null)
                throw new NullPointerException();
            if (size > 0 && (tab = table) != null) {
                int mc = modCount;
                for (int i = 0; i < tab.length; ++i) {
                    for (Node<K,V> e = tab[i]; e != null; e = e.next)
                        action.accept(e);
                }
                if (modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }
    }

    // Overrides of JDK8 Map extension methods

    @Override
    public V getOrDefault(Object key, V defaultValue) {
        Node<K,V> e;
        return (e = getNode(hash(key), key)) == null ? defaultValue : e.value;
    }

    @Override
    public V putIfAbsent(K key, V value) {
        return putVal(hash(key), key, value, true, true);
    }

    @Override
    public boolean remove(Object key, Object value) {
        return removeNode(hash(key), key, value, true, true) != null;
    }

    @Override
    public boolean replace(K key, V oldValue, V newValue) {
        Node<K,V> e; V v;
        if ((e = getNode(hash(key), key)) != null &&
            ((v = e.value) == oldValue || (v != null && v.equals(oldValue)))) {
            e.value = newValue;
            afterNodeAccess(e);
            return true;
        }
        return false;
    }

    @Override
    public V replace(K key, V value) {
        Node<K,V> e;
        if ((e = getNode(hash(key), key)) != null) {
            V oldValue = e.value;
            e.value = value;
            afterNodeAccess(e);
            return oldValue;
        }
        return null;
    }

    @Override
    public V computeIfAbsent(K key,
                             Function<? super K, ? extends V> mappingFunction) {
        if (mappingFunction == null)
            throw new NullPointerException();
        int hash = hash(key);
        Node<K,V>[] tab; Node<K,V> first; int n, i;
        int binCount = 0;
        TreeNode<K,V> t = null;
        Node<K,V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0)
            n = (tab = resize()).length;
        if ((first = tab[i = (n - 1) & hash]) != null) {
            if (first instanceof TreeNode)
                old = (t = (TreeNode<K,V>)first).getTreeNode(hash, key);
            else {
                Node<K,V> e = first; K k;
                do {
                    if (e.hash == hash &&
                        ((k = e.key) == key || (key != null && key.equals(k)))) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
            V oldValue;
            if (old != null && (oldValue = old.value) != null) {
                afterNodeAccess(old);
                return oldValue;
            }
        }
        V v = mappingFunction.apply(key);
        if (v == null) {
            return null;
        } else if (old != null) {
            old.value = v;
            afterNodeAccess(old);
            return v;
        }
        else if (t != null)
            t.putTreeVal(this, tab, hash, key, v);
        else {
            tab[i] = newNode(hash, key, v, first);
            if (binCount >= TREEIFY_THRESHOLD - 1)
                treeifyBin(tab, hash);
        }
        ++modCount;
        ++size;
        afterNodeInsertion(true);
        return v;
    }

    public V computeIfPresent(K key,
                              BiFunction<? super K, ? super V, ? extends V> remappingFunction) {
        if (remappingFunction == null)
            throw new NullPointerException();
        Node<K,V> e; V oldValue;
        int hash = hash(key);
        if ((e = getNode(hash, key)) != null &&
            (oldValue = e.value) != null) {
            V v = remappingFunction.apply(key, oldValue);
            if (v != null) {
                e.value = v;
                afterNodeAccess(e);
                return v;
            }
            else
                removeNode(hash, key, null, false, true);
        }
        return null;
    }

    @Override
    public V compute(K key,
                     BiFunction<? super K, ? super V, ? extends V> remappingFunction) {
        if (remappingFunction == null)
            throw new NullPointerException();
        int hash = hash(key);
        Node<K,V>[] tab; Node<K,V> first; int n, i;
        int binCount = 0;
        TreeNode<K,V> t = null;
        Node<K,V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0)
            n = (tab = resize()).length;
        if ((first = tab[i = (n - 1) & hash]) != null) {
            if (first instanceof TreeNode)
                old = (t = (TreeNode<K,V>)first).getTreeNode(hash, key);
            else {
                Node<K,V> e = first; K k;
                do {
                    if (e.hash == hash &&
                        ((k = e.key) == key || (key != null && key.equals(k)))) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
        }
        V oldValue = (old == null) ? null : old.value;
        V v = remappingFunction.apply(key, oldValue);
        if (old != null) {
            if (v != null) {
                old.value = v;
                afterNodeAccess(old);
            }
            else
                removeNode(hash, key, null, false, true);
        }
        else if (v != null) {
            if (t != null)
                t.putTreeVal(this, tab, hash, key, v);
            else {
                tab[i] = newNode(hash, key, v, first);
                if (binCount >= TREEIFY_THRESHOLD - 1)
                    treeifyBin(tab, hash);
            }
            ++modCount;
            ++size;
            afterNodeInsertion(true);
        }
        return v;
    }

    @Override
    public V merge(K key, V value,
                   BiFunction<? super V, ? super V, ? extends V> remappingFunction) {
        if (value == null)
            throw new NullPointerException();
        if (remappingFunction == null)
            throw new NullPointerException();
        int hash = hash(key);
        Node<K,V>[] tab; Node<K,V> first; int n, i;
        int binCount = 0;
        TreeNode<K,V> t = null;
        Node<K,V> old = null;
        if (size > threshold || (tab = table) == null ||
            (n = tab.length) == 0)
            n = (tab = resize()).length;
        if ((first = tab[i = (n - 1) & hash]) != null) {
            if (first instanceof TreeNode)
                old = (t = (TreeNode<K,V>)first).getTreeNode(hash, key);
            else {
                Node<K,V> e = first; K k;
                do {
                    if (e.hash == hash &&
                        ((k = e.key) == key || (key != null && key.equals(k)))) {
                        old = e;
                        break;
                    }
                    ++binCount;
                } while ((e = e.next) != null);
            }
        }
        if (old != null) {
            V v;
            if (old.value != null)
                v = remappingFunction.apply(old.value, value);
            else
                v = value;
            if (v != null) {
                old.value = v;
                afterNodeAccess(old);
            }
            else
                removeNode(hash, key, null, false, true);
            return v;
        }
        if (value != null) {
            if (t != null)
                t.putTreeVal(this, tab, hash, key, value);
            else {
                tab[i] = newNode(hash, key, value, first);
                if (binCount >= TREEIFY_THRESHOLD - 1)
                    treeifyBin(tab, hash);
            }
            ++modCount;
            ++size;
            afterNodeInsertion(true);
        }
        return value;
    }

    @Override
    public void forEach(BiConsumer<? super K, ? super V> action) {
        Node<K,V>[] tab;
        if (action == null)
            throw new NullPointerException();
        if (size > 0 && (tab = table) != null) {
            int mc = modCount;
            for (int i = 0; i < tab.length; ++i) {
                for (Node<K,V> e = tab[i]; e != null; e = e.next)
                    action.accept(e.key, e.value);
            }
            if (modCount != mc)
                throw new ConcurrentModificationException();
        }
    }

    @Override
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function) {
        Node<K,V>[] tab;
        if (function == null)
            throw new NullPointerException();
        if (size > 0 && (tab = table) != null) {
            int mc = modCount;
            for (int i = 0; i < tab.length; ++i) {
                for (Node<K,V> e = tab[i]; e != null; e = e.next) {
                    e.value = function.apply(e.key, e.value);
                }
            }
            if (modCount != mc)
                throw new ConcurrentModificationException();
        }
    }

    /* ------------------------------------------------------------ */
    // Cloning and serialization

    /**
     * Returns a shallow copy of this <tt>HashMap</tt> instance: the keys and
     * values themselves are not cloned.
     *
     * @return a shallow copy of this map
     */
    @SuppressWarnings("unchecked")
    @Override
    public Object clone() {
        HashMap<K,V> result;
        try {
            result = (HashMap<K,V>)super.clone();
        } catch (CloneNotSupportedException e) {
            // this shouldn't happen, since we are Cloneable
            throw new InternalError(e);
        }
        result.reinitialize();
        result.putMapEntries(this, false);
        return result;
    }

    // These methods are also used when serializing HashSets
    final float loadFactor() { return loadFactor; }
    final int capacity() {
        return (table != null) ? table.length :
            (threshold > 0) ? threshold :
            DEFAULT_INITIAL_CAPACITY;
    }

    /**
     * Save the state of the <tt>HashMap</tt> instance to a stream (i.e.,
     * serialize it).
     *
     * @serialData The <i>capacity</i> of the HashMap (the length of the
     *             bucket array) is emitted (int), followed by the
     *             <i>size</i> (an int, the number of key-value
     *             mappings), followed by the key (Object) and value (Object)
     *             for each key-value mapping.  The key-value mappings are
     *             emitted in no particular order.
     */
    private void writeObject(java.io.ObjectOutputStream s)
        throws IOException {
        int buckets = capacity();
        // Write out the threshold, loadfactor, and any hidden stuff
        s.defaultWriteObject();
        s.writeInt(buckets);
        s.writeInt(size);
        internalWriteEntries(s);
    }

    /**
     * Reconstitute the {@code HashMap} instance from a stream (i.e.,
     * deserialize it).
     */
    private void readObject(java.io.ObjectInputStream s)
        throws IOException, ClassNotFoundException {
        // Read in the threshold (ignored), loadfactor, and any hidden stuff
        s.defaultReadObject();
        reinitialize();
        if (loadFactor <= 0 || Float.isNaN(loadFactor))
            throw new InvalidObjectException("Illegal load factor: " +
                                             loadFactor);
        s.readInt();                // Read and ignore number of buckets
        int mappings = s.readInt(); // Read number of mappings (size)
        if (mappings < 0)
            throw new InvalidObjectException("Illegal mappings count: " +
                                             mappings);
        else if (mappings > 0) { // (if zero, use defaults)
            // Size the table using given load factor only if within
            // range of 0.25...4.0
            float lf = Math.min(Math.max(0.25f, loadFactor), 4.0f);
            float fc = (float)mappings / lf + 1.0f;
            int cap = ((fc < DEFAULT_INITIAL_CAPACITY) ?
                       DEFAULT_INITIAL_CAPACITY :
                       (fc >= MAXIMUM_CAPACITY) ?
                       MAXIMUM_CAPACITY :
                       tableSizeFor((int)fc));
            float ft = (float)cap * lf;
            threshold = ((cap < MAXIMUM_CAPACITY && ft < MAXIMUM_CAPACITY) ?
                         (int)ft : Integer.MAX_VALUE);

            // Check Map.Entry[].class since it's the nearest public type to
            // what we're actually creating.
            SharedSecrets.getJavaOISAccess().checkArray(s, Map.Entry[].class, cap);
            @SuppressWarnings({"rawtypes","unchecked"})
            Node<K,V>[] tab = (Node<K,V>[])new Node[cap];
            table = tab;

            // Read the keys and values, and put the mappings in the HashMap
            for (int i = 0; i < mappings; i++) {
                @SuppressWarnings("unchecked")
                    K key = (K) s.readObject();
                @SuppressWarnings("unchecked")
                    V value = (V) s.readObject();
                putVal(hash(key), key, value, false, false);
            }
        }
    }

    /* ------------------------------------------------------------ */
    // iterators

    abstract class HashIterator {
        Node<K,V> next;        // next entry to return
        Node<K,V> current;     // current entry
        int expectedModCount;  // for fast-fail
        int index;             // current slot

        HashIterator() {
            expectedModCount = modCount;
            Node<K,V>[] t = table;
            current = next = null;
            index = 0;
            if (t != null && size > 0) { // advance to first entry
                do {} while (index < t.length && (next = t[index++]) == null);
            }
        }

        public final boolean hasNext() {
            return next != null;
        }

        final Node<K,V> nextNode() {
            Node<K,V>[] t;
            Node<K,V> e = next;
            if (modCount != expectedModCount)
                throw new ConcurrentModificationException();
            if (e == null)
                throw new NoSuchElementException();
            if ((next = (current = e).next) == null && (t = table) != null) {
                do {} while (index < t.length && (next = t[index++]) == null);
            }
            return e;
        }

        public final void remove() {
            Node<K,V> p = current;
            if (p == null)
                throw new IllegalStateException();
            if (modCount != expectedModCount)
                throw new ConcurrentModificationException();
            current = null;
            K key = p.key;
            removeNode(hash(key), key, null, false, false);
            expectedModCount = modCount;
        }
    }

    final class KeyIterator extends HashIterator
        implements Iterator<K> {
        public final K next() { return nextNode().key; }
    }

    final class ValueIterator extends HashIterator
        implements Iterator<V> {
        public final V next() { return nextNode().value; }
    }

    final class EntryIterator extends HashIterator
        implements Iterator<Map.Entry<K,V>> {
        public final Map.Entry<K,V> next() { return nextNode(); }
    }

    /* ------------------------------------------------------------ */
    // spliterators

    static class HashMapSpliterator<K,V> {
        final HashMap<K,V> map;
        Node<K,V> current;          // current node
        int index;                  // current index, modified on advance/split
        int fence;                  // one past last index
        int est;                    // size estimate
        int expectedModCount;       // for comodification checks

        HashMapSpliterator(HashMap<K,V> m, int origin,
                           int fence, int est,
                           int expectedModCount) {
            this.map = m;
            this.index = origin;
            this.fence = fence;
            this.est = est;
            this.expectedModCount = expectedModCount;
        }

        final int getFence() { // initialize fence and size on first use
            int hi;
            if ((hi = fence) < 0) {
                HashMap<K,V> m = map;
                est = m.size;
                expectedModCount = m.modCount;
                Node<K,V>[] tab = m.table;
                hi = fence = (tab == null) ? 0 : tab.length;
            }
            return hi;
        }

        public final long estimateSize() {
            getFence(); // force init
            return (long) est;
        }
    }

    static final class KeySpliterator<K,V>
        extends HashMapSpliterator<K,V>
        implements Spliterator<K> {
        KeySpliterator(HashMap<K,V> m, int origin, int fence, int est,
                       int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public KeySpliterator<K,V> trySplit() {
            int hi = getFence(), lo = index, mid = (lo + hi) >>> 1;
            return (lo >= mid || current != null) ? null :
                new KeySpliterator<>(map, lo, index = mid, est >>>= 1,
                                        expectedModCount);
        }

        public void forEachRemaining(Consumer<? super K> action) {
            int i, hi, mc;
            if (action == null)
                throw new NullPointerException();
            HashMap<K,V> m = map;
            Node<K,V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = (tab == null) ? 0 : tab.length;
            }
            else
                mc = expectedModCount;
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<K,V> p = current;
                current = null;
                do {
                    if (p == null)
                        p = tab[i++];
                    else {
                        action.accept(p.key);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }

        public boolean tryAdvance(Consumer<? super K> action) {
            int hi;
            if (action == null)
                throw new NullPointerException();
            Node<K,V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null)
                        current = tab[index++];
                    else {
                        K k = current.key;
                        current = current.next;
                        action.accept(k);
                        if (map.modCount != expectedModCount)
                            throw new ConcurrentModificationException();
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return (fence < 0 || est == map.size ? Spliterator.SIZED : 0) |
                Spliterator.DISTINCT;
        }
    }

    static final class ValueSpliterator<K,V>
        extends HashMapSpliterator<K,V>
        implements Spliterator<V> {
        ValueSpliterator(HashMap<K,V> m, int origin, int fence, int est,
                         int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public ValueSpliterator<K,V> trySplit() {
            int hi = getFence(), lo = index, mid = (lo + hi) >>> 1;
            return (lo >= mid || current != null) ? null :
                new ValueSpliterator<>(map, lo, index = mid, est >>>= 1,
                                          expectedModCount);
        }

        public void forEachRemaining(Consumer<? super V> action) {
            int i, hi, mc;
            if (action == null)
                throw new NullPointerException();
            HashMap<K,V> m = map;
            Node<K,V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = (tab == null) ? 0 : tab.length;
            }
            else
                mc = expectedModCount;
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<K,V> p = current;
                current = null;
                do {
                    if (p == null)
                        p = tab[i++];
                    else {
                        action.accept(p.value);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }

        public boolean tryAdvance(Consumer<? super V> action) {
            int hi;
            if (action == null)
                throw new NullPointerException();
            Node<K,V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null)
                        current = tab[index++];
                    else {
                        V v = current.value;
                        current = current.next;
                        action.accept(v);
                        if (map.modCount != expectedModCount)
                            throw new ConcurrentModificationException();
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return (fence < 0 || est == map.size ? Spliterator.SIZED : 0);
        }
    }

    static final class EntrySpliterator<K,V>
        extends HashMapSpliterator<K,V>
        implements Spliterator<Map.Entry<K,V>> {
        EntrySpliterator(HashMap<K,V> m, int origin, int fence, int est,
                         int expectedModCount) {
            super(m, origin, fence, est, expectedModCount);
        }

        public EntrySpliterator<K,V> trySplit() {
            int hi = getFence(), lo = index, mid = (lo + hi) >>> 1;
            return (lo >= mid || current != null) ? null :
                new EntrySpliterator<>(map, lo, index = mid, est >>>= 1,
                                          expectedModCount);
        }

        public void forEachRemaining(Consumer<? super Map.Entry<K,V>> action) {
            int i, hi, mc;
            if (action == null)
                throw new NullPointerException();
            HashMap<K,V> m = map;
            Node<K,V>[] tab = m.table;
            if ((hi = fence) < 0) {
                mc = expectedModCount = m.modCount;
                hi = fence = (tab == null) ? 0 : tab.length;
            }
            else
                mc = expectedModCount;
            if (tab != null && tab.length >= hi &&
                (i = index) >= 0 && (i < (index = hi) || current != null)) {
                Node<K,V> p = current;
                current = null;
                do {
                    if (p == null)
                        p = tab[i++];
                    else {
                        action.accept(p);
                        p = p.next;
                    }
                } while (p != null || i < hi);
                if (m.modCount != mc)
                    throw new ConcurrentModificationException();
            }
        }

        public boolean tryAdvance(Consumer<? super Map.Entry<K,V>> action) {
            int hi;
            if (action == null)
                throw new NullPointerException();
            Node<K,V>[] tab = map.table;
            if (tab != null && tab.length >= (hi = getFence()) && index >= 0) {
                while (current != null || index < hi) {
                    if (current == null)
                        current = tab[index++];
                    else {
                        Node<K,V> e = current;
                        current = current.next;
                        action.accept(e);
                        if (map.modCount != expectedModCount)
                            throw new ConcurrentModificationException();
                        return true;
                    }
                }
            }
            return false;
        }

        public int characteristics() {
            return (fence < 0 || est == map.size ? Spliterator.SIZED : 0) |
                Spliterator.DISTINCT;
        }
    }

    /* ------------------------------------------------------------ */
    // LinkedHashMap support


    /*
     * The following package-protected methods are designed to be
     * overridden by LinkedHashMap, but not by any other subclass.
     * Nearly all other internal methods are also package-protected
     * but are declared final, so can be used by LinkedHashMap, view
     * classes, and HashSet.
     */

    // Create a regular (non-tree) node
    Node<K,V> newNode(int hash, K key, V value, Node<K,V> next) {
        return new Node<>(hash, key, value, next);
    }

    // For conversion from TreeNodes to plain nodes
    Node<K,V> replacementNode(Node<K,V> p, Node<K,V> next) {
        return new Node<>(p.hash, p.key, p.value, next);
    }

    // Create a tree bin node
    TreeNode<K,V> newTreeNode(int hash, K key, V value, Node<K,V> next) {
        return new TreeNode<>(hash, key, value, next);
    }

    // For treeifyBin
    TreeNode<K,V> replacementTreeNode(Node<K,V> p, Node<K,V> next) {
        return new TreeNode<>(p.hash, p.key, p.value, next);
    }

    /**
     * Reset to initial default state.  Called by clone and readObject.
     */
    void reinitialize() {
        table = null;
        entrySet = null;
        keySet = null;
        values = null;
        modCount = 0;
        threshold = 0;
        size = 0;
    }

    // Callbacks to allow LinkedHashMap post-actions
    void afterNodeAccess(Node<K,V> p) { }
    void afterNodeInsertion(boolean evict) { }
    void afterNodeRemoval(Node<K,V> p) { }

    // Called only from writeObject, to ensure compatible ordering.
    void internalWriteEntries(java.io.ObjectOutputStream s) throws IOException {
        Node<K,V>[] tab;
        if (size > 0 && (tab = table) != null) {
            for (int i = 0; i < tab.length; ++i) {
                for (Node<K,V> e = tab[i]; e != null; e = e.next) {
                    s.writeObject(e.key);
                    s.writeObject(e.value);
                }
            }
        }
    }

    /* ------------------------------------------------------------ */
    // Tree bins

    /**
     * Entry for Tree bins. Extends LinkedHashMap.Entry (which in turn
     * extends Node) so can be used as extension of either regular or
     * linked node.
     */
    static final class TreeNode<K,V> extends LinkedHashMap.Entry<K,V> {
        TreeNode<K,V> parent;  // red-black tree links
        TreeNode<K,V> left;
        TreeNode<K,V> right;
        TreeNode<K,V> prev;    // needed to unlink next upon deletion
        boolean red;
        TreeNode(int hash, K key, V val, Node<K,V> next) {
            super(hash, key, val, next);
        }

        /**
         * Returns root of tree containing this node.
         */
        final TreeNode<K,V> root() {
            for (TreeNode<K,V> r = this, p;;) {
                if ((p = r.parent) == null)
                    return r;
                r = p;
            }
        }

        /**
         * Ensures that the given root is the first node of its bin.<br>
         * 如果当前索引位置的头节点不是root节点, 则将root的上一个节点和下一个节点进行关联,<br>
         * 将root放到头节点的位置, 原头节点放在root的next节点上
         */
        static <K,V> void moveRootToFront(Node<K,V>[] tab, TreeNode<K,V> root) {
            int n;
            if (root != null && tab != null && (n = tab.length) > 0) {
                int index = (n - 1) & root.hash;
                TreeNode<K,V> first = (TreeNode<K,V>)tab[index];
                if (root != first) { // 如果root节点不是该索引位置的头节点
                    Node<K,V> rn;
                    tab[index] = root; // 将该索引位置的头节点赋值为root根节点
                    TreeNode<K,V> rp = root.prev;
                    // 如果root节点的下一个节点不为空, 则将root节点的下一个节点的prev属性设置为root节点的上一个节点
                    if ((rn = root.next) != null)
                        ((TreeNode<K,V>)rn).prev = rp;
                    // 如果root节点的上一个节点不为空, 则将root节点的上一个节点的next属性设置为root节点的下一个节点
                    if (rp != null)
                        rp.next = rn;
                    // 如果原头节点不为空, 则将原头节点的prev属性设置为root节点
                    if (first != null)
                        first.prev = root;
                    root.next = first; // 将root节点的next节点设置为原头节点
                    root.prev = null; // 将root节点的prev节点设置为null
                }
                assert checkInvariants(root); // 检查树是否正常
            }
        }

        /**
         * Finds the node starting at root p with the given hash and key.
         * The kc argument caches comparableClassFor(key) upon first use
         * comparing keys.
         */
        final TreeNode<K,V> find(int h, Object k, Class<?> kc) {
            TreeNode<K,V> p = this;
            do {
                int ph, dir; K pk;
                TreeNode<K,V> pl = p.left, pr = p.right, q;
                // 传入的hash值小于p节点的hash值，则往p节点的左边遍历
                if ((ph = p.hash) > h)
                    p = pl;
                // 传入的hash值大于p节点的hash值，则往p节点的右边遍历
                else if (ph < h)
                    p = pr;
                // 传入的hash值、key与p节点的hash值、key相等时，则p为目标节点，返回p节点
                else if ((pk = p.key) == k || (k != null && k.equals(pk)))
                    return p;
                // p节点的左节点为空，则向右遍历
                else if (pl == null)
                    p = pr;
                // p节点的右节点为空，则向左遍历
                else if (pr == null)
                    p = pl;
                // 如果传入的key(k)所属的类实现了Comparable接口，则将传入的k与p.key进行比较
                else if ((kc != null ||
                          (kc = comparableClassFor(k)) != null) && // 此行结果为true代表k实现Comparable接口
                         (dir = compareComparables(kc, k, pk)) != 0) // k<pk则dir<0, k>pk则dir>0
                    p = (dir < 0) ? pl : pr; // dir<0向左遍历，dir>0向右遍历
                // 代码走到此处，说明key所属类没有实现Comparable接口，直接指定向p的右边遍历
                else if ((q = pr.find(h, k, kc)) != null)
                    return q;
                // 代码走到此处，说明上一个向右遍历(pr.find(h, k, kc))无结果，因此直接向左遍历
                else
                    p = pl;
            } while (p != null);
            return null;
        }

        /**
         * Calls find for root node.
         */
        final TreeNode<K,V> getTreeNode(int h, Object k) {
            return ((parent != null) ? root() : this).find(h, k, null);
        }

        /**
         * Tie-breaking utility for ordering insertions when equal
         * hashCodes and non-comparable. We don't require a total
         * order, just a consistent insertion rule to maintain
         * equivalence across rebalancings. Tie-breaking further than
         * necessary simplifies testing a bit.
         */
        // 用于哈希值相同并且无法比较时进行比较，只是一个一致性的插入规则，用来维护重定位的等价性
        static int tieBreakOrder(Object a, Object b) {
            int d;
            if (a == null || b == null ||
                (d = a.getClass().getName().
                 compareTo(b.getClass().getName())) == 0)
                d = (System.identityHashCode(a) <= System.identityHashCode(b) ?
                     -1 : 1);
            return d;
        }

        /**
         * Forms tree of the nodes linked from this node.
         * @return root of tree
         */
        final void treeify(Node<K,V>[] tab) {
            TreeNode<K,V> root = null;
            for (TreeNode<K,V> x = this, next; x != null; x = next) { // this即为调用此方法的树节点
                next = (TreeNode<K,V>)x.next; // next赋值为x的next节点
                x.left = x.right = null; // 将x的左右节点设置为空
                if (root == null) { // 如果还没有根节点，则将x构造为根节点
                    x.parent = null; // 根节点没有父节点
                    x.red = false; // 根节点必须为黑色
                    root = x; // 将x设置为根节点
                }
                else {
                    K k = x.key;
                    int h = x.hash;
                    Class<?> kc = null;
                    // 如果当前节点x不是根节点，则从根节点开始查找属于该节点的位置
                    for (TreeNode<K,V> p = root;;) {
                        int dir, ph;
                        K pk = p.key;
                        // 如果调用节点的hash值小于p节点的hash值，dir赋值为-1，表示向p的左边查找
                        if ((ph = p.hash) > h)
                            dir = -1;
                        // 如果调用节点的hash值大于p节点的hash值，dir赋值为1，表示向p的右边查找
                        else if (ph < h)
                            dir = 1;
                        // 如果k没有实现Comparable接口 或者 x节点的key和p节点的key相等
                        else if ((kc == null &&
                                  (kc = comparableClassFor(k)) == null) ||
                                 (dir = compareComparables(kc, k, pk)) == 0)
                            // 使用定义的一套规则来比较x节点和p节点的大小，用来决定向左还是向右查找
                            dir = tieBreakOrder(k, pk);

                        TreeNode<K,V> xp = p;
                        if ((p = (dir <= 0) ? p.left : p.right) == null) {
                            x.parent = xp;
                            if (dir <= 0)
                                xp.left = x;
                            else
                                xp.right = x;
                            // 进行红黑树的插入平衡(通过左旋、右旋和改变节点颜色来保证当前树符合红黑树的要求)
                            root = balanceInsertion(root, x);
                            break;
                        }
                    }
                }
            }
            // 如果root节点不在table索引位置的头结点, 则将其调整为头结点
            moveRootToFront(tab, root);
        }

        /**
         * Returns a list of non-TreeNodes replacing those linked from
         * this node.
         */
        final Node<K,V> untreeify(HashMap<K,V> map) {
            Node<K,V> hd = null, tl = null; // hd指向头结点, tl指向尾节点
            // 从调用该方法的节点, 即链表的头结点开始遍历, 将所有节点全转为链表节点
            for (Node<K,V> q = this; q != null; q = q.next) {
                // 调用replacementNode方法构建链表节点
                Node<K,V> p = map.replacementNode(q, null);
                // 如果tl为null, 则代表当前节点为第一个节点, 将hd赋值为该节点
                if (tl == null)
                    hd = p;
                // 否则, 将尾节点的next属性设置为当前节点p
                else
                    tl.next = p;
                tl = p; // 每次都将tl节点指向当前节点, 即尾节点
            }
            return hd; // 返回转换后的链表的头结点
        }

        /**
         * Tree version of putVal.
         * 红黑树插入会同时维护原来的链表属性, 即原来的next属性
         */
        final TreeNode<K,V> putTreeVal(HashMap<K,V> map, Node<K,V>[] tab,
                                       int h, K k, V v) {
            Class<?> kc = null;
            boolean searched = false;
            // 查找根节点，索引位置的头节点并不一定为红黑树的根节点
            TreeNode<K,V> root = (parent != null) ? root() : this;
            for (TreeNode<K,V> p = root;;) { // 将根节点赋值给p，开始遍历
                int dir, ph; K pk;
                // 如果传入的hash值h，小于p节点的hash值，则将dir赋值为-1，表示向p的左边遍历
                if ((ph = p.hash) > h)
                    dir = -1;
                // 如果传入的hash值h，大于p节点的hash值，则将dir赋值为1，表示向p的右边遍历
                else if (ph < h)
                    dir = 1;
                // 走到这说明传入的hash等于p节点的hash，先判断传入的key与p.key是否相等，相等则p为目标节点，返回p节点
                else if ((pk = p.key) == k || (k != null && k.equals(pk)))
                    return p;
                // 如果传入的key(K)没有实现Comparable接口，或者实现了Comparable接口并且根据compareTo方法判断key与p.key相等
                else if ((kc == null &&
                          (kc = comparableClassFor(k)) == null) ||
                         (dir = compareComparables(kc, k, pk)) == 0) {
                    // 则进行一次树的节点查找
                    if (!searched) {
                        TreeNode<K,V> q, ch;
                        searched = true; // 只有第一次会进行此遍历
                        // 从p节点的左节点和右节点分别根据hash、key、与key.Class调用find方法进行查找，如果查找到节点则直接返回
                        if (((ch = p.left) != null &&
                             (q = ch.find(h, k, kc)) != null) ||
                            ((ch = p.right) != null &&
                             (q = ch.find(h, k, kc)) != null))
                            return q;
                    }
                    // 否则，使用定义的一套规则来比较k和p.key的大小，从而决定向左还是向右查找
                    dir = tieBreakOrder(k, pk);
                }

                TreeNode<K,V> xp = p; // xp赋值为x的父节点，中间变量，用于下面给x的父节点赋值
                // dir<=0则向p左边查找，否则向p右边查找，如果为null，则表示该位置即为x的目标位置
                if ((p = (dir <= 0) ? p.left : p.right) == null) {
                    // 走进来，则表示已经找到x的位置，只需将x放到该位置即可
                    Node<K,V> xpn = xp.next;
                    // 创建新的节点x，将新节点x插入x的父节点xp与xp.next之间
                    TreeNode<K,V> x = map.newTreeNode(h, k, v, xpn);
                    if (dir <= 0)
                        xp.left = x; // 如果dir<=0，则表示x为xp的左节点
                    else
                        xp.right = x; // 否则，表示x为xp的右节点
                    xp.next = x; // 将xp的next节点设置为x
                    x.parent = x.prev = xp; // 将x的父节点、prev节点设置为xp
                    // 如果xpn不为空，则将xpn的prev节点设置x，与上文(map.newTreeNode)的x.next=xpn对应
                    if (xpn != null)
                        ((TreeNode<K,V>)xpn).prev = x;
                    moveRootToFront(tab, balanceInsertion(root, x)); // 进行红黑树的插入平衡调整
                    return null;
                }
            }
        }


        /**
         * <pre>
         * Removes the given node, that must be present before this call.
         * This is messier than typical red-black deletion code because we
         * cannot swap the contents of an interior node with a leaf
         * successor that is pinned by "next" pointers that are accessible
         * independently during traversal. So instead we swap the tree
         * linkages. If the current tree appears to have too few nodes,
         * the bin is converted back to a plain bin. (The test triggers
         * somewhere between 2 and 6 nodes, depending on tree structure).
         *
         *  如果table为空或者length为0直接返回。
         *  根据hash值和length-1位于运算计算出索引的位置。
         *  将索引位置的头结点赋值给first和root，removeTreeNode方法是被将要移除的节点node调用，因此removeTreeNode方法里的this即为将要被移除的节点node，将node的next节点赋值给succ节点，prev节点赋值给pred节点。
         *  如果node节点的prev节点为空，则代表要被移除的node节点为头结点，则将table索引位置的值和first节点的值赋值为node的next节点（succ节点）即可。
         *  否则将node的prev节点（pred节点）的next节点设置为node的next节点（succ节点），如果succ节点不为空，则将succ的prev节点设置为pred，与前面对应（TreeNode链表的移除，见开头第8点）。
         *  如果进行到此first节点为空，则代表该索引位置已经没有节点则直接返回。
         *  如果root的父节点不为空，则将root赋值为根结点（root在上面被赋值为索引位置的头结点，索引位置的头节点并不一定为红黑树的根结点）。
         *  通过root节点来判断此红黑树是否太小，如果太小则转为链表节点并返回（转链表后就无需再进行下面的红黑树处理），链表维护部分到此结束，此前的代码说明了，红黑树在进行移除的同时也会维护链表结构，之后的代码为红黑树的移除节点处理。
         *  上面已经说了this为将要被移除的node节点，将p节点赋值为将要被移除的node节点（则此时p节点就是我们要移除的节点），pl赋值为node的左节点, pr赋值为node的右节点（方法的命令见开头第6点），replacement变量用来存储将要替换掉被移除的node节点。
         *  如果p的左节点和右节点都不为空时，s节点赋值为p的右节点；向s的左节点一直向左查找, 直到叶子节点，跳出循环时，s为叶子节点；交换p节点和s节点（叶子节点）的颜色（此文下面的所有操作都是为了实现将p节点和s节点进行位置调换，因此此处先将颜色替换）；sr赋值为s节点的右节点，pp节点赋值为p节点的父节点（命令规律见文章开头第6点）。
         *  PS：下面的第一次调整和第二次调整是将p节点和s节点进行了位置调换，然后找出要替换掉p节点的replacement；第三次调整是将replacement节点覆盖掉p节点；这部分的代码逻辑比较不容易理解透，建议自己动手画图模拟。（下文图解1即为这三次调整的例子）
         *  进行第一次调整：如果p节点的右节点即为叶子节点，将p的父节点赋值为s，将s的右节点赋值为p即可；否则，将p的父节点赋值为s的父节点sp，并判断sp是否为空，如果不为空，并判断s是sp的左节点还是右节点，将s节点替换为p节点；将s的右节点赋值为p节点的右节点pr，如果pr不为空则将pr的父节赋值为s节点。
         *  进行第二次调整：将p节点的左节点清空（上文pl已经保存了该节点）；将p节点的右节点赋值为s的右节点sr，如果sr不为空，则将sr的父节点赋值为p节点；将s节点的左节点赋值为p的左节点pl，如果pl不为空，则将p左节点的父节点赋值为s节点；将s的父节点赋值为p的父节点pp，如果pp为空，则p节点为root节点，此时交换后s成为新的root节点，将root赋值为s节点；如果p不为root节点，并且p是父节点的左节点，将p父节点的左节点赋值为s节点；如果p不为root节点，并且p是父节点的右节点，将p父节点的右节点赋值为s节点；如果sr不为空，将replacement赋值为sr节点，否则赋值为p节点（为什么sr是replacement的首选，p为备选？见解释1）。
         *  承接第10点的判断，第10点~第12点为p的左右节点都不为空的情况需要进行的处理；如果p的左节点不为空，右节点为空，将replacement赋值为p的左节点即可；如果p的右节点不为空，左节点为空，将replacement赋值为p的右节点即可；如果p的左右节点都为空，即p为叶子节点, 将replacement赋值为p节点本身。
         *  进行第三次调整：如果p节点不是replacement（即p不是叶子节点），将replacement的父节点赋值为p的父节点，同事赋值给pp节点；如果pp为空（p节点没有父节点），即p为root节点，则将root节点赋值为replacement节点即可；如果p节点不是root节点，并且p节点为父节点的左节点，则将p父节点的左节点赋值为replacement节点；如果p节点不是root节点，并且p节点为父节点的右节点，则将p父节点的右节点赋值为replacement节点；p节点的位置已经被完整的替换为replacement节点, 将p节点清空。
         *  如果p节点不为红色则进行红黑树删除平衡调整（如果删除的节点是红色则不会破坏红黑树的平衡无需调整，见文末的解释2）。
         *  如果p节点为叶子节点，则简单的将p节点移除：将pp赋值为p节点的父节点，将p的parent节点设置为空，如果p的父节点pp存在，如果p节点为父节点的左节点，则将父节点的左节点赋值为空，如果p节点为父节点的右节点，则将父节点的右节点赋值为空。
         *  如果movable为true，则调用moveRootToFront方法（见上文代码块8）将root节点移到索引位置的头结点。
         *  </pre>
         */
        final void removeTreeNode(HashMap<K,V> map, Node<K,V>[] tab,
                                  boolean movable) {
            // ==========  链表的处理 start ==========
            int n;
            if (tab == null || (n = tab.length) == 0) // 表为空，或者表长度为空，直接返回
                return;
            // 根据节点hash值计算该节点的索引位置
            int index = (n - 1) & hash;
            // first、node均赋值为索引位置的头节点
            TreeNode<K,V> first = (TreeNode<K,V>)tab[index], root = first, rl;
            // succ赋值为this.next节点，pred赋值为this.prev节点
            TreeNode<K,V> succ = (TreeNode<K,V>)next, pred = prev;
            // 如果this.prev为空，将table索引位置的节点、first节点赋值为this.next节点
            if (pred == null)
                tab[index] = first = succ;
            // 否则，将this.prev.next节点赋值为this.next节点（移除this.prev节点与this的链表关系）
            else
                pred.next = succ;
            // 如果this.next不为空，则将this.next.prev赋值为this.prev（移除this.next.prev节点与this的链表关系，与上步对应）
            if (succ != null)
                succ.prev = pred;
            // 如果first为空，代表该索引位置无任何节点，或者只有头节点且头节点被移除，即当前已无节点，直接返回
            if (first == null)
                return;
            // 如果root的父节点不为空，则将root赋值为根节点
            if (root.parent != null)
                root = root.root();
            // 通过root节点判断此红黑树是否太小，如果是则调用untreeify方法转为链表节点并返回
            // （转链表后无需进行下面的红黑树处理，直接返回）
            if (root == null || root.right == null ||
                (rl = root.left) == null || rl.left == null) {
                tab[index] = first.untreeify(map);  // too small
                return;
            }
            // ==========  链表的处理 end ==========
            // ==========  红黑树的处理 start ==========
            // p、pl、pr分别赋值为this、this.left、this.right
            TreeNode<K,V> p = this, pl = left, pr = right, replacement;
            // node的左节点和右节点都不为空时
            if (pl != null && pr != null) {
                TreeNode<K,V> s = pr, sl;
                // 向左遍历查找，直到找到叶子节点s，跳出循环
                while ((sl = s.left) != null) // find successor
                    s = sl;
                boolean c = s.red; s.red = p.red; p.red = c; // swap colors 交换p节点和s节点(叶子节点)的颜色
                TreeNode<K,V> sr = s.right; // sr赋值为叶子节点s的右子节点s.right
                TreeNode<K,V> pp = p.parent; // pp赋值为该节点的父节点this.parent
                // ========== 第一次调整 start ==========
                // 如果p为s的父节点
                if (s == pr) { // p was s's direct parent
                    p.parent = s; // 将p的父节点赋值为s
                    s.right = p; // 将s的右节点赋值为p
                }
                else {
                    TreeNode<K,V> sp = s.parent;
                    if ((p.parent = sp) != null) { // 将p的父节点赋值为s的父节点, 如果sp不为空
                        if (s == sp.left) // 如果s节点为左节点
                            sp.left = p; // 则将s的父节点的左节点赋值为p节点
                        else // 如果s节点为右节点
                            sp.right = p; // 则将s的父节点的右节点赋值为p节点
                    }
                    if ((s.right = pr) != null) // s的右节点赋值为p节点的右节点
                        pr.parent = s; // p节点的右节点的父节点赋值为s
                }
                // ========== 第二次调整 start ==========
                p.left = null;
                if ((p.right = sr) != null) // 将p节点的右节点赋值为s的右节点, 如果sr不为空
                    sr.parent = p; // 则将s右节点的父节点赋值为p节点
                if ((s.left = pl) != null) // 将s节点的左节点赋值为p的左节点, 如果pl不为空
                    pl.parent = s; // 则将p左节点的父节点赋值为s节点
                if ((s.parent = pp) == null) // 将s的父节点赋值为p的父节点pp, 如果pp为空
                    root = s; // 则p节点为root节点, 此时交换后s成为新的root节点
                else if (p == pp.left) // 如果p不为root节点, 并且p是父节点的左节点
                    pp.left = s; // 将p父节点的左节点赋值为s节点
                else // 如果p不为root节点, 并且p是父节点的右节点
                    pp.right = s; // 将p父节点的右节点赋值为s节点
                if (sr != null)
                    replacement = sr; // 寻找replacement节点(用来替换掉p节点)
                else
                    replacement = p; // 寻找replacement节点
            }
            else if (pl != null) // 如果p的左节点不为空,右节点为空,replacement节点为p的左节点
                replacement = pl;
            else if (pr != null) // 如果p的右节点不为空,左节点为空,replacement节点为p的右节点
                replacement = pr;
            else // 如果p的左右节点都为空, 即p为叶子节点, 替换节点为p节点本身
                replacement = p;
            // ========== 第三次调整 start ==========
            if (replacement != p) { // 如果p节点不是叶子节点
                // 将replacement节点的父节点赋值为p节点的父节点, 同时赋值给pp节点
                TreeNode<K,V> pp = replacement.parent = p.parent;
                if (pp == null) // 如果p节点没有父节点, 即p为root节点
                    root = replacement; // 则将root节点赋值为replacement节点即可
                else if (p == pp.left) // 如果p节点不是root节点, 并且p节点为父节点的左节点
                    pp.left = replacement; // 则将p父节点的左节点赋值为替换节点
                else // 如果p节点不是root节点, 并且p节点为父节点的右节点
                    pp.right = replacement; // 则将p父节点的右节点赋值为替换节点
                // p节点的位置已经被完整的替换为替换节点, 将p节点清空, 以便垃圾收集器回收
                p.left = p.right = p.parent = null;
            }
            // 如果p节点不为红色则进行红黑树删除平衡调整
            // (如果删除的节点是红色则不会破坏红黑树的平衡无需调整)
            TreeNode<K,V> r = p.red ? root : balanceDeletion(root, replacement);

            if (replacement == p) {  // detach // 如果p节点为叶子节点, 则简单的将p节点去除即可
                TreeNode<K,V> pp = p.parent; // pp赋值为p节点的父节点
                p.parent = null; // 将p的parent节点设置为空
                if (pp != null) { // 如果p的父节点存在
                    if (p == pp.left) // 如果p节点为父节点的左节点
                        pp.left = null; // 则将父节点的左节点赋值为空
                    else if (p == pp.right) // 如果p节点为父节点的右节点
                        pp.right = null; // 则将父节点的右节点赋值为空
                }
            }
            if (movable)
                moveRootToFront(tab, r); // 将root节点移到索引位置的头结点
        }

        /* ****************************************************************************************
         *                                    removeTreeNode图解                                  *
         * ****************************************************************************************
         * 最复杂的情况：                                                                         *
         *                                                                                        *
         *       pp             |     pp              |      pp             |      pp             *
         *         \            |                     |        \            |        \            *
         *          p           |         s           |         s           |         s           *
         *         / \          |          \          |        / \          |        / \          *
         *       pl   pr        |           pr        |       pl  pr        |       pl  pr        *
         *            /         |           /         |           /         |           /         *
         *          sp          |          sp         |          sp         |          sp         *
         *         /            |          /          |          /          |          /          *
         *        s             |         p           |         p           |         sr          *
         *         \            |        /            |          \          |                     *
         *          sr          |       pl  sr        |           sr        |          p          *
         *                      |                     | (replacement = sr)  |                     *
         *    第一次调整前      |    第一次调整后     |    第二次调整后     |    第三次调整后     *
         * ********** ********** ********** ********** ********** ********** ********** ***********/

        /*  ****************************************************************************************
         *   解释1：为什么sr是replacement的首选，p为备选？
         *  ****************************************************************************************
         *   解析：首先我们看sr是什么？从代码中可以看到sr第一次被赋值时，是在s节点进行了向左穷遍历结束后，
         *   因此此时s节点是没有左节点的，sr即为s节点的右节点。而从上面的三次调整我们知道，p节点已经跟
         *   s节点进行了位置调换，所以此时sr其实是p节点的右节点，并且p节点没有左节点，因此要移除p节点，
         *   只需要将p节点的右节点sr覆盖掉p节点即可，因此sr是replacement的首选，如果sr为空，则代表p节点
         *   为叶子节点，此时将p节点清空即可。
         */

        /*  ****************************************************************************************
         *   解释2：关于红黑树的平衡调整？
         *  ****************************************************************************************
         *   红黑树是一种自平衡二叉树，拥有优秀的查询和插入/删除性能，广泛应用于关联数组。
         *   对比AVL树，AVL要求每个结点的左右子树的高度之差的绝对值（平衡因子）最多为1，而红黑树通过适当的
         *   放低该条件（红黑树限制从根到叶子的最长的可能路径不多于最短的可能路径的两倍长，结果是这个树
         *   大致上是平衡的），以此来减少插入/删除时的平衡调整耗时，从而获取更好的性能，而这虽然会导致红黑树
         *   的查询会比AVL稍慢，但相比插入/删除时获取的时间，这个付出在大多数情况下显然是值得的。
         *   在HashMap中的应用：HashMap在进行插入和删除时有可能会触发红黑树的插入平衡调整（balanceInsertion方法）
         *   或删除平衡调整（balanceDeletion ）方法，调整的方式主要有以下手段：左旋转（rotateLeft方法）、
         *   右旋转（rotateRight方法）、改变节点颜色（x.red = false、x.red = true），进行调整的原因是为了
         *   维持红黑树的数据结构。
         */

        /**
         * Splits nodes in a tree bin into lower and upper tree bins,
         * or untreeifies if now too small. Called only from resize;
         * see above discussion about split bits and indices.
         * <pre>
         * 以调用此方法的节点开始，遍历整个红黑树节点（此处实际是遍历的链表节点，上文提过，红黑树节点也会同时维护链表结构）。
         * 	如果e的hash值与老表的容量（为一串只有1个为2的二进制数，例如16为0000 0000 0001 0000）进行位与运算为0，则说明e节点扩容后的索引位置跟老表的索引位置一样（见下文例子1），进行链表拼接操作：如果loTail为空，代表该节点为第一个节点，则将loHead赋值为该节点；否则将节点添加在loTail后面，并将loTail赋值为新增的节点，并统计原索引位置的节点个数。
         * 	如果e的hash值与老表的容量（为一串只有1个为2的二进制数，例如16为0000 0000 0001 0000）进行位与运算为1，则说明e节点扩容后的索引位置为：老表的索引位置＋oldCap（见例子1），进行链表拼接操作：如果hiTail为空，代表该节点为第一个节点，则将hiHead赋值为该节点；否则将节点添加在hiTail后面，并将hiTail赋值为新增的节点，并统计索引位置为原索引+oldCap的节点个数。
         * 	如果原索引位置的节点不为空：如果当该索引位置节点数<=6个，调用untreeify方法（见下文代码块11）将红黑树节点转为链表节点；否则将原索引位置的节点设置为对应的头结点（即loHead结点），如果判断hiHead不为空则代表原来的红黑树（老表的红黑树由于节点被分到两个位置）已经被改变，需要重新构建新的红黑树，以loHead为根结点，调用treeify方法（见上文代码块7）构建新的红黑树。
         * 	如果索引位置为原索引+oldCap的节点不为空：如果当该索引位置节点数<=6个，调用untreeify方法（见下文代码块11）将红黑树节点转为链表节点；否则将索引位置为原索引+oldCap的节点设置为对应的头结点（即hiHead结点），如果判断loHead不为空则代表原来的红黑树（老表的红黑树由于节点被分到两个位置）已经被改变，需要重新构建新的红黑树，以hiHead为根结点，调用treeify方法（见上文代码块7）构建新的红黑树。
         * ---------------------
         * 作者：程序员囧辉
         * 来源：CSDN
         * 原文：https://blog.csdn.net/v123411739/article/details/78996181
         * 版权声明：本文为博主原创文章，转载请附上博文链接！
         * </pre>
         * @param map the map
         * @param tab the table for recording bin heads
         * @param index the index of the table being split
         * @param bit the bit of hash to split on
         */
        final void split(HashMap<K,V> map, Node<K,V>[] tab, int index, int bit) {
            TreeNode<K,V> b = this; // 拿到调用此方法的节点并赋值给节点b
            // Relink into lo and hi lists, preserving order
            TreeNode<K,V> loHead = null, loTail = null; // 存储跟原索引位置相同的节点
            TreeNode<K,V> hiHead = null, hiTail = null; // 存储索引位置为:原索引+oldCap的节点
            int lc = 0, hc = 0;
            for (TreeNode<K,V> e = b, next; e != null; e = next) { // 从b节点开始遍历
                next = (TreeNode<K,V>)e.next; // next赋值为e的下个节点
                e.next = null; // 同时将老表的节点设置为空，以便垃圾收集器回收
                // 如果e的hash值与老表的容量进行与运算为0,则扩容后的索引位置跟老表的索引位置一样
                if ((e.hash & bit) == 0) {
                    if ((e.prev = loTail) == null) // 如果loTail为空, 代表该节点为第一个节点
                        loHead = e; // 则将loHead赋值为第一个节点
                    else
                        loTail.next = e;  // 否则将节点添加在loTail后面
                    loTail = e; // 并将loTail赋值为新增的节点
                    ++lc; // 统计原索引位置的节点个数
                }
                // 如果e的hash值与老表的容量进行与运算不为0,则扩容后的索引位置为:老表的索引位置＋oldCap
                else {
                    if ((e.prev = hiTail) == null) // 如果hiTail为空, 代表该节点为第一个节点
                        hiHead = e; // 则将hiHead赋值为第一个节点
                    else
                        hiTail.next = e; // 否则将节点添加在hiTail后面
                    hiTail = e; // 并将hiTail赋值为新增的节点
                    ++hc; // 统计索引位置为原索引+oldCap的节点个数
                }
            }

            if (loHead != null) { // 原索引位置的节点不为空
                if (lc <= UNTREEIFY_THRESHOLD) // 节点个数少于6个则将红黑树转为链表结构
                    tab[index] = loHead.untreeify(map);
                else {
                    tab[index] = loHead; // 将原索引位置的节点设置为对应的头结点
                    // hiHead不为空则代表原来的红黑树(老表的红黑树由于节点被分到两个位置)
                    // 已经被改变, 需要重新构建新的红黑树
                    if (hiHead != null) // (else is already treeified)
                        loHead.treeify(tab); // 以loHead为根结点, 构建新的红黑树
                }
            }
            if (hiHead != null) { // 索引位置为原索引+oldCap的节点不为空
                if (hc <= UNTREEIFY_THRESHOLD) // 节点个数少于6个则将红黑树转为链表结构
                    tab[index + bit] = hiHead.untreeify(map);
                else {
                    tab[index + bit] = hiHead; // 将索引位置为原索引+oldCap的节点设置为对应的头结点
                    // loHead不为空则代表原来的红黑树(老表的红黑树由于节点被分到两个位置)
                    // 已经被改变, 需要重新构建新的红黑树
                    if (loHead != null)
                        hiHead.treeify(tab);  // 以hiHead为根结点, 构建新的红黑树
                }
            }
        }

        /* ------------------------------------------------------------ */
        // Red-black tree methods, all adapted from CLR

        static <K,V> TreeNode<K,V> rotateLeft(TreeNode<K,V> root,
                                              TreeNode<K,V> p) {
            TreeNode<K,V> r, pp, rl;
            if (p != null && (r = p.right) != null) {
                if ((rl = p.right = r.left) != null)
                    rl.parent = p;
                if ((pp = r.parent = p.parent) == null)
                    (root = r).red = false;
                else if (pp.left == p)
                    pp.left = r;
                else
                    pp.right = r;
                r.left = p;
                p.parent = r;
            }
            return root;
        }

        static <K,V> TreeNode<K,V> rotateRight(TreeNode<K,V> root,
                                               TreeNode<K,V> p) {
            TreeNode<K,V> l, pp, lr;
            if (p != null && (l = p.left) != null) {
                if ((lr = p.left = l.right) != null)
                    lr.parent = p;
                if ((pp = l.parent = p.parent) == null)
                    (root = l).red = false;
                else if (pp.right == p)
                    pp.right = l;
                else
                    pp.left = l;
                l.right = p;
                p.parent = l;
            }
            return root;
        }

        static <K,V> TreeNode<K,V> balanceInsertion(TreeNode<K,V> root,
                                                    TreeNode<K,V> x) {
            x.red = true;
            for (TreeNode<K,V> xp, xpp, xppl, xppr;;) {
                if ((xp = x.parent) == null) {
                    x.red = false;
                    return x;
                }
                else if (!xp.red || (xpp = xp.parent) == null)
                    return root;
                if (xp == (xppl = xpp.left)) {
                    if ((xppr = xpp.right) != null && xppr.red) {
                        xppr.red = false;
                        xp.red = false;
                        xpp.red = true;
                        x = xpp;
                    }
                    else {
                        if (x == xp.right) {
                            root = rotateLeft(root, x = xp);
                            xpp = (xp = x.parent) == null ? null : xp.parent;
                        }
                        if (xp != null) {
                            xp.red = false;
                            if (xpp != null) {
                                xpp.red = true;
                                root = rotateRight(root, xpp);
                            }
                        }
                    }
                }
                else {
                    if (xppl != null && xppl.red) {
                        xppl.red = false;
                        xp.red = false;
                        xpp.red = true;
                        x = xpp;
                    }
                    else {
                        if (x == xp.left) {
                            root = rotateRight(root, x = xp);
                            xpp = (xp = x.parent) == null ? null : xp.parent;
                        }
                        if (xp != null) {
                            xp.red = false;
                            if (xpp != null) {
                                xpp.red = true;
                                root = rotateLeft(root, xpp);
                            }
                        }
                    }
                }
            }
        }

        static <K,V> TreeNode<K,V> balanceDeletion(TreeNode<K,V> root,
                                                   TreeNode<K,V> x) {
            for (TreeNode<K,V> xp, xpl, xpr;;)  {
                if (x == null || x == root)
                    return root;
                else if ((xp = x.parent) == null) {
                    x.red = false;
                    return x;
                }
                else if (x.red) {
                    x.red = false;
                    return root;
                }
                else if ((xpl = xp.left) == x) {
                    if ((xpr = xp.right) != null && xpr.red) {
                        xpr.red = false;
                        xp.red = true;
                        root = rotateLeft(root, xp);
                        xpr = (xp = x.parent) == null ? null : xp.right;
                    }
                    if (xpr == null)
                        x = xp;
                    else {
                        TreeNode<K,V> sl = xpr.left, sr = xpr.right;
                        if ((sr == null || !sr.red) &&
                            (sl == null || !sl.red)) {
                            xpr.red = true;
                            x = xp;
                        }
                        else {
                            if (sr == null || !sr.red) {
                                if (sl != null)
                                    sl.red = false;
                                xpr.red = true;
                                root = rotateRight(root, xpr);
                                xpr = (xp = x.parent) == null ?
                                    null : xp.right;
                            }
                            if (xpr != null) {
                                xpr.red = (xp == null) ? false : xp.red;
                                if ((sr = xpr.right) != null)
                                    sr.red = false;
                            }
                            if (xp != null) {
                                xp.red = false;
                                root = rotateLeft(root, xp);
                            }
                            x = root;
                        }
                    }
                }
                else { // symmetric
                    if (xpl != null && xpl.red) {
                        xpl.red = false;
                        xp.red = true;
                        root = rotateRight(root, xp);
                        xpl = (xp = x.parent) == null ? null : xp.left;
                    }
                    if (xpl == null)
                        x = xp;
                    else {
                        TreeNode<K,V> sl = xpl.left, sr = xpl.right;
                        if ((sl == null || !sl.red) &&
                            (sr == null || !sr.red)) {
                            xpl.red = true;
                            x = xp;
                        }
                        else {
                            if (sl == null || !sl.red) {
                                if (sr != null)
                                    sr.red = false;
                                xpl.red = true;
                                root = rotateLeft(root, xpl);
                                xpl = (xp = x.parent) == null ?
                                    null : xp.left;
                            }
                            if (xpl != null) {
                                xpl.red = (xp == null) ? false : xp.red;
                                if ((sl = xpl.left) != null)
                                    sl.red = false;
                            }
                            if (xp != null) {
                                xp.red = false;
                                root = rotateRight(root, xp);
                            }
                            x = root;
                        }
                    }
                }
            }
        }

        /**
         * Recursive invariant check
         */
        static <K,V> boolean checkInvariants(TreeNode<K,V> t) {
            TreeNode<K,V> tp = t.parent, tl = t.left, tr = t.right,
                tb = t.prev, tn = (TreeNode<K,V>)t.next;
            if (tb != null && tb.next != t)
                return false;
            if (tn != null && tn.prev != t)
                return false;
            if (tp != null && t != tp.left && t != tp.right)
                return false;
            // 如果左节点不为空，左节点的hash值不能大于t节点的hash值
            if (tl != null && (tl.parent != t || tl.hash > t.hash))
                return false;
            // 如果右节点不为空，右节点的hash值不能小于t节点的hash值
            if (tr != null && (tr.parent != t || tr.hash < t.hash))
                return false;
            // 如果当前节点为红色, 则该节点的左右节点都不能为红色
            if (t.red && tl != null && tl.red && tr != null && tr.red)
                return false;
            // 遍历校验左子树
            if (tl != null && !checkInvariants(tl))
                return false;
            // 遍历校验右子树
            if (tr != null && !checkInvariants(tr))
                return false;
            return true;
        }
    }


    /* ********************************************************************************************************
     *
     * 例子1: 扩容后，节点重hash为什么只可能分布在原索引位置与原索引+oldCap位置？
     *
     *   假设老表的容量为16，即oldCap=16，则新表容量为16*2=32，假设节点1的hash值为0000 0000 0000 0000 0000 1111 0000 1010，
     * 节点2的hash值为0000 0000 0000 0000 0000 1111 0001 1010，则节点1和节点2在老表的索引位置计算如下图计算1，由于老表的
     * 长度限制，节点1和节点2的索引位置只取决于节点hash值的最后4位。再看计算2，计算2为新表的索引计算，可以知道如果两个
     * 节点在老表的索引位置相同，则新表的索引位置只取决于节点hash值倒数第5位的值，而此位置的值刚好为老表的容量值16，此时
     * 节点在新表的索引位置只有两种情况：原索引位置和原索引+oldCap位置（在此例中即为10和10+16=26）。由于结果只取决于节点
     * hash值的倒数第5位，而此位置的值刚好为老表的容量值16，因此此时新表的索引位置的计算可以替换为计算3，直接使用节点的
     * hash值与老表的容量16进行位于运算，如果结果为0则该节点在新表的索引位置为原索引位置，否则该节点在新表的索引位置为原
     * 索引+oldCap位置。
     *
     * 计算1：
     *    老表容量-1：0000 0000 0000 0000 0000 0000 0000 1111
     *    节点1：     0000 0000 0000 0000 0000 1111 0000 1010   ->   0000 0000 0000 0000 0000 0000 0000 1010
     *    节点2：     0000 0000 0000 0000 0000 1111 0001 1010   ->   0000 0000 0000 0000 0000 0000 0000 1010
     *
     * 计算2：
     *    新表容量-1：0000 0000 0000 0000 0000 0000 0001 1111
     *    节点1：     0000 0000 0000 0000 0000 1111 0000 1010   ->   0000 0000 0000 0000 0000 0000 0000 1010
     *    节点2：     0000 0000 0000 0000 0000 1111 0001 1010   ->   0000 0000 0000 0000 0000 0000 0001 1010
     *
     * 计算3：
     *    老表容量：  0000 0000 0000 0000 0000 0000 0001 0000
     *    节点1：     0000 0000 0000 0000 0000 1111 0000 1010   ->   0000 0000 0000 0000 0000 0000 0000 0000
     *    节点2：     0000 0000 0000 0000 0000 1111 0001 1010   ->   0000 0000 0000 0000 0000 0000 0001 0000
     *
     **********************************************************************************************************/

    /*
     * HashMap和Hashtable的区别：
     *    1.HashMap允许key和value为null，Hashtable不允许。
	 *    2.HashMap的默认初始容量为16，Hashtable为11。
	 *    3.HashMap的扩容为原来的2倍，Hashtable的扩容为原来的2倍加1。
	 *    4.HashMap是非线程安全的，Hashtable是线程安全的。
	 *    5.HashMap的hash值重新计算过，Hashtable直接使用hashCode。
	 *    6.HashMap去掉了Hashtable中的contains方法。
	 *    7.HashMap继承自AbstractMap类，Hashtable继承自Dictionary类。
     */

    /*
     * 总结：
     *   1. HashMap的底层是个Node数组（Node<K,V>[] table），在数组的具体索引位置，
     *      如果存在多个节点，则可能是以链表或红黑树的形式存在。
	 *   2 .增加、删除、查找键值对时，定位到哈希桶数组的位置是很关键的一步，源码中是通过下面3个操作来完成这一步：
	 *          1）拿到key的hashCode值；
	 *          2）将hashCode的高位参与运算，重新计算hash值；
	 *          3）将计算出来的hash值与(table.length - 1)进行&运算。
	 *   3. HashMap的默认初始容量（capacity）是16，capacity必须为2的幂次方；默认负载因子（load factor）是0.75；
	 *      实际能存放的节点个数（threshold，即触发扩容的阈值）= capacity * load factor。
	 *   4. HashMap在触发扩容后，阈值会变为原来的2倍，并且会进行重hash，重hash后索引位置index的节点的新分布位置
	 *      最多只有两个：原索引位置或原索引+oldCap位置。例如capacity为16，索引位置5的节点扩容后，只可能分布在
	 *      新报索引位置5和索引位置21（5+16）。
	 *   5. 导致HashMap扩容后，同一个索引位置的节点重hash最多分布在两个位置的根本原因是：
	 *          1）table的长度始终为2的n次方；
	 *          2）索引位置的计算方法为“(table.length - 1) & hash”。HashMap扩容是一个比较耗时的操作，
	 *          定义HashMap时尽量给个接近的初始容量值。
	 *   6. HashMap有threshold属性和loadFactor属性，但是没有capacity属性。初始化时，如果传了初始化容量值，
	 *      该值是存在threshold变量，并且Node数组是在第一次put时才会进行初始化，初始化时会将此时的threshold值
	 *      作为新表的capacity值，然后用capacity和loadFactor计算新表的真正threshold值。
	 *   7. 当同一个索引位置的节点在增加后达到9个时，会触发链表节点（Node）转红黑树节点（TreeNode，间接继承Node），
	 *      转成红黑树节点后，其实链表的结构还存在，通过next属性维持。链表节点转红黑树节点的具体方法为源码中的
	 *      treeifyBin(Node<K,V>[] tab, int hash)方法。
	 *   8. 当同一个索引位置的节点在移除后达到6个时，并且该索引位置的节点为红黑树节点，会触发红黑树节点转链表节点。
	 *      红黑树节点转链表节点的具体方法为源码中的untreeify(HashMap<K,V> map)方法。
	 *   9. HashMap在JDK1.8之后不再有死循环的问题，JDK1.8之前存在死循环的根本原因是在扩容后同一索引位置的节点顺序会反掉。
	 *   10.HashMap是非线程安全的，在并发场景下使用ConcurrentHashMap来代替。
     */
}
